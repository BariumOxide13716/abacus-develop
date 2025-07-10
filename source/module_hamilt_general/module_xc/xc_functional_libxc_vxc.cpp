#ifdef USE_LIBXC

#include "xc_functional.h"
#include "xc_functional_libxc.h"
#include "module_elecstate/module_charge/charge.h"
#include "module_base/global_variable.h"
#include "module_parameter/parameter.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"

#include <xc.h>
#include <list>

#include <vector>

std::tuple<double,double,ModuleBase::matrix> XC_Functional_Libxc::v_xc_libxc(		// Peize Lin update for nspin==4 at 2023.01.14
        const std::vector<int> &func_id,
        const int &nrxx, // number of real-space grid
        const double &omega, // volume of cell
        const double tpiba,
        const Charge* const chr,
        const std::map<int, double>* scaling_factor)
{
    ModuleBase::TITLE("XC_Functional_Libxc","v_xc_libxc");
    ModuleBase::timer::tick("XC_Functional_Libxc","v_xc_libxc");
    std::cout << "do I reach v_xc_libxc?" << std::endl;
    const int nspin =
        (PARAM.inp.nspin == 1 || ( PARAM.inp.nspin ==4 && !PARAM.globalv.domag && !PARAM.globalv.domag_z))
        ? 1 : 2;

    //----------------------------------------------------------
    // xc_func_type is defined in Libxc package
    // to understand the usage of xc_func_type,
    // use can check on website, for example:
    // https://www.tddft.org/programs/libxc/manual/libxc-5.1.x/
    //----------------------------------------------------------

    std::vector<xc_func_type> funcs = XC_Functional_Libxc::init_func(
        /* func_id = */ func_id, 
        /* xc_polarized = */ (1==nspin) ? XC_UNPOLARIZED : XC_POLARIZED);

    const bool is_gga = [&funcs]()
    {
        for( xc_func_type &func : funcs )
        {
            switch( func.info->family )
            {
                case XC_FAMILY_GGA:
                case XC_FAMILY_HYB_GGA:
                    return true;
            }
        }
        return false;
    }();

    // converting rho
    std::vector<double> rho;
    std::vector<double> amag;
    if(1==nspin || 2==PARAM.inp.nspin)
    {
        rho = XC_Functional_Libxc::convert_rho(nspin, nrxx, chr);
    }
    else
    {
        std::tuple<std::vector<double>,std::vector<double>> rho_amag = XC_Functional_Libxc::convert_rho_amag_nspin4(nspin, nrxx, chr);
        rho = std::get<0>(std::move(rho_amag));
        amag = std::get<1>(std::move(rho_amag));
    }

    std::vector<std::vector<ModuleBase::Vector3<double>>> gdr;
    std::vector<double> sigma;
    if(is_gga)
    {
        gdr = XC_Functional_Libxc::cal_gdr(nspin, nrxx, rho, tpiba, chr);
        sigma = XC_Functional_Libxc::convert_sigma(gdr);
    }

    double etxc = 0.0;
    double vtxc = 0.0;
    ModuleBase::matrix v(nspin,nrxx);

    for( xc_func_type &func : funcs )
    {
        // jiyy add for threshold
        constexpr double rho_threshold = 1E-6;
        constexpr double grho_threshold = 1E-10;

        xc_func_set_dens_threshold(&func, rho_threshold);

        // sgn for threshold mask
        const std::vector<double> sgn = XC_Functional_Libxc::cal_sgn(rho_threshold, grho_threshold, func, nspin, nrxx, rho, sigma);

        std::vector<double> exc   ( nrxx                    );
        std::vector<double> vrho  ( nrxx * nspin            );
        std::vector<double> vsigma( nrxx * ((1==nspin)?1:3) );
        switch( func.info->family )
        {
            case XC_FAMILY_LDA:
                // call Libxc function: xc_lda_exc_vxc
                xc_lda_exc_vxc( &func, nrxx, rho.data(),
                    exc.data(), vrho.data() );
                break;
            case XC_FAMILY_GGA:
            case XC_FAMILY_HYB_GGA:
                // call Libxc function: xc_gga_exc_vxc
                xc_gga_exc_vxc( &func, nrxx, rho.data(), sigma.data(),
                    exc.data(), vrho.data(), vsigma.data() );
                break;
            default:
                throw std::domain_error("func.info->family ="+std::to_string(func.info->family)
                    +" unfinished in "+std::string(__FILE__)+" line "+std::to_string(__LINE__));
                break;
        }

        // added by jghan, 2024-10-10
        double factor = 1.0;
        if( scaling_factor == nullptr ) { ;
        } else
        {
            auto pair_factor = scaling_factor->find(func.info->number);
            if( pair_factor != scaling_factor->end() ) { factor = pair_factor->second;
}
        }

        // time factor is added by jghan, 2024-10-10
        etxc += XC_Functional_Libxc::convert_etxc(nspin, nrxx, sgn, rho, exc) * factor;
        const std::pair<double,ModuleBase::matrix> vtxc_v = XC_Functional_Libxc::convert_vtxc_v(
            func, nspin, nrxx,
            sgn, rho, gdr,
            vrho, vsigma,
            tpiba, chr);
        vtxc += std::get<0>(vtxc_v) * factor;
        v += std::get<1>(vtxc_v) * factor;
    } // end for( xc_func_type &func : funcs )

    if(4==PARAM.inp.nspin)
    {
        v = XC_Functional_Libxc::convert_v_nspin4(nrxx, chr, amag, v);
    }

    //-------------------------------------------------
    // for MPI, reduce the exchange-correlation energy
    //-------------------------------------------------
    #ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
    #endif

    etxc *= omega / chr->rhopw->nxyz;
    vtxc *= omega / chr->rhopw->nxyz;

    XC_Functional_Libxc::finish_func(funcs);

    ModuleBase::timer::tick("XC_Functional_Libxc","v_xc_libxc");
    return std::make_tuple( etxc, vtxc, std::move(v) );
}


//the interface to libxc xc_mgga_exc_vxc(xc_func,n,rho,grho,laplrho,tau,e,v1,v2,v3,v4)
//xc_func : LIBXC data type, contains information on xc functional
//n: size of array, nspin*nnr
//rho,grho,laplrho: electron density, its gradient and laplacian
//tau(kin_r): kinetic energy density
//e: energy density
//v1-v4: derivative of energy density w.r.t rho, gradient, laplacian and tau
//v1 and v2 are combined to give v; v4 goes into vofk

//XC_POLARIZED, XC_UNPOLARIZED: internal flags used in LIBXC, denote the polarized(nspin=1) or unpolarized(nspin=2) calculations, definition can be found in xc.h from LIBXC

// [etxc, vtxc, v, vofk] = XC_Functional::v_xc(...)
std::tuple<double,double,ModuleBase::matrix,ModuleBase::matrix> XC_Functional_Libxc::v_xc_meta(
    const std::vector<int> &func_id,
    const int &nrxx, // number of real-space grid
    const double &omega, // volume of cell
    const double tpiba,
    const Charge* const chr)
{
    ModuleBase::TITLE("XC_Functional_Libxc","v_xc_meta");
    ModuleBase::timer::tick("XC_Functional_Libxc","v_xc_meta");


    std::cout << " hello from XC_Functional_Libxc::v_xc_meta " << std::endl;
    int n_neg_dens_a = 0;
    int n_neg_dens_b = 0;
    int n_neg_tau_a = 0;
    int n_neg_tau_b = 0;
    int n_discard_a = 0;
    int n_discard_b = 0;
    int i_reach_2 = 0;
    int i_reach_3 = 0;
    int n_neg_dens = 0;
    int n_zero_sigma = 0;
    int n_neg_tau = 0;
    int n_grid_print = 6;
    double e2 = 2.0;

        /// use the max_discard_rhoa, max_discard_rhob, ir_max_discard_rhoa, ir_max_discard_rhob
        /// to store the maximum discarded alpha and beta density points.
        /// Do similar things for sigma_aa, sigma_bb, tau_a and tau_b.
        double max_discard_rhoa = 0;
        double max_discard_rhob = 0;
        int ir_max_discard_rhoa = 0;
        int ir_max_discard_rhob = 0;
        double max_discard_sigma_aa = 0;
        double max_discard_sigma_bb = 0;
        int ir_max_discard_sigma_aa = 0;
        int ir_max_discard_sigma_bb = 0;
        double max_discard_tau_a = 0;
        double max_discard_tau_b = 0;
        int ir_max_discard_tau_a = 0;
        int ir_max_discard_tau_b = 0;

    //output of the subroutine
    double etxc = 0.0;
    double vtxc = 0.0;
    ModuleBase::matrix v(PARAM.inp.nspin,nrxx);
    ModuleBase::matrix vofk(PARAM.inp.nspin,nrxx);

    //----------------------------------------------------------
    // xc_func_type is defined in Libxc package
    // to understand the usage of xc_func_type,
    // use can check on website, for example:
    // https://www.tddft.org/programs/libxc/manual/libxc-5.1.x/
    //----------------------------------------------------------

    const int nspin = PARAM.inp.nspin;
    std::vector<xc_func_type> funcs = XC_Functional_Libxc::init_func(
        /* func_id = */ func_id, 
        /* xc_polarized = */ (1==nspin) ? XC_UNPOLARIZED:XC_POLARIZED);

    const std::vector<double> rho = XC_Functional_Libxc::convert_rho(nspin, nrxx, chr);
    const std::vector<std::vector<ModuleBase::Vector3<double>>> gdr
        = XC_Functional_Libxc::cal_gdr(nspin, nrxx, rho, tpiba, chr);
    const std::vector<double> sigma = XC_Functional_Libxc::convert_sigma(gdr);

    //converting kin_r
    std::vector<double> kin_r;
    kin_r.resize(nrxx*nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for( int is=0; is<nspin; ++is )
    {
        for( int ir=0; ir<nrxx; ++ir )
        {
            kin_r[ir*nspin+is] = chr->kin_r[is][ir] / 2.0;
        }
    }

    std::vector<double> exc    ( nrxx                    );
    std::vector<double> vrho   ( nrxx * nspin            );
    std::vector<double> vsigma ( nrxx * ((1==nspin)?1:3) );
    std::vector<double> vtau   ( nrxx * nspin            );
    std::vector<double> vlapl  ( nrxx * nspin            );

    double rho_th  = PARAM.inp.dft_thre_density; // density threshold
    double grho_th = PARAM.inp.dft_thre_density_gradient; // gradient threshold
    double tau_th  = PARAM.inp.dft_thre_kin_ene_density; // kinetic energy density threshold
    // sgn for threshold mask
    std::vector<double> sgn( nrxx * nspin);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for(int i = 0; i < nrxx * nspin; ++i)
    {
        sgn[i] = 1.0;
    }

    if(nspin == 1)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for( int ir=0; ir<nrxx; ++ir )
        {
            if ( rho[ir]<0 ){
                n_neg_dens += 1;
            }
            if ( sqrt(std::abs(sigma[ir]))<grho_th ){
                n_zero_sigma += 1;
            }
            if ( kin_r[ir]<tau_th){
                n_neg_tau += 1;
            }

            //if ( rho[ir]<rho_th || sqrt(std::abs(sigma[ir]))<grho_th || std::abs(kin_r[ir])<tau_th)
            if ( rho[ir]<0 || sqrt(std::abs(sigma[ir]))<grho_th  || sigma[ir] < 0.0 || kin_r[ir]<tau_th)
            {
                sgn[ir] = 0.0;
            }
        }
        std::cout << "number of negative density points:   "
                  << n_neg_dens << "/" << nrxx << std::endl; 
        std::cout << "number of zero sigma points:         "
                  << n_zero_sigma << "/" << nrxx << std::endl; 
        std::cout << "number of negative tau points:       "
                  << n_neg_tau << "/" << nrxx << std::endl; 
    }
    else
    {

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif

        for( int ir=0; ir<nrxx; ++ir )
        {
            if ( rho[ir*2] < 0.0  ){
                n_neg_dens_a += 1;
            }
            if ( rho[ir*2+1] < 0.0  ){
                n_neg_dens_b += 1;
            }
            if ( kin_r[ir*2] < 0.0  ){
                n_neg_tau_a += 1;
            }
            if ( kin_r[ir*2+1] < 0.0  ){
                n_neg_tau_b += 1;
            }
            if ( rho[ir*2]<rho_th || sqrt(std::abs(sigma[ir*3]))<grho_th || sigma[ir*3] < 0.0 || kin_r[ir*2]<tau_th) {
            //if ( rho[ir*2]<rho_th || sqrt(std::abs(sigma[ir*3]))<grho_th || std::abs(kin_r[ir*2])<tau_th) {
                sgn[ir*2] = 0.0;
                sgn[ir*2+1] = 0.0;
                n_discard_a += 1;
                if ( rho[ir*2]<rho_th && rho[ir*2] > max_discard_rhoa )
                {
                    max_discard_rhoa = rho[ir*2];
                    ir_max_discard_rhoa = ir;
                }
                if ( sqrt(std::abs(sigma[ir*3]))<grho_th && sigma[ir*3] > max_discard_sigma_aa )
                {
                    max_discard_sigma_aa = sigma[ir*3];
                    ir_max_discard_sigma_aa = ir;
                }
                if ( kin_r[ir*2]<tau_th && kin_r[ir*2] > max_discard_tau_a )
                {
                    max_discard_tau_a = kin_r[ir*2];
                    ir_max_discard_tau_a = ir;
                }
}
            if ( rho[ir*2+1]<rho_th || sqrt(std::abs(sigma[ir*3+2]))<grho_th || sigma[ir*3+1] < 0.0|| kin_r[ir*2+1]<tau_th) {
            //if ( rho[ir*2+1]<rho_th || sqrt(std::abs(sigma[ir*3+2]))<grho_th || std::abs(kin_r[ir*2+1])<tau_th) {
                sgn[ir*2] = 0.0;
                sgn[ir*2+1] = 0.0;
                n_discard_b += 1;
                if ( rho[ir*2+1]<rho_th && rho[ir*2+1] > max_discard_rhob )
                {
                    max_discard_rhob = rho[ir*2+1];
                    ir_max_discard_rhob = ir;
                }
                if ( sqrt(std::abs(sigma[ir*3+2]))<grho_th && sigma[ir*3+2] > max_discard_sigma_bb )
                {
                    max_discard_sigma_bb = sigma[ir*3+2];
                    ir_max_discard_sigma_bb = ir;
                }
                if ( kin_r[ir*2+1]<tau_th && kin_r[ir*2+1] > max_discard_tau_b )
                {
                    max_discard_tau_b = kin_r[ir*2+1];
                    ir_max_discard_tau_b = ir;
                }
}
            /// if rho[ir*2] < rho_th or rho[ir*2+1], set sgn[ir*2] and sgn[ir*2+1] to 0.0
            //if ( rho[ir*2] < rho[ir*2+1])
            //{
            //    sgn[ir*2] = sgn[ir*2+1] = 0.0;
            //}
        }
        std::cout << "number of negative alpha and beta density points:         "
                  << n_neg_dens_a << "/" << nrxx << " "
                  << n_neg_dens_b << "/" << nrxx << std::endl; 
        std::cout << "number of negative alpha and beta kinetic-density points: "
                  << n_neg_tau_a << "/" << nrxx << " "
                  << n_neg_tau_b << "/" << nrxx << std::endl; 
        std::cout << "number of discarded alpha and beta points:                "
                  << n_discard_a << "/" << nrxx << " "
                  << n_discard_b << "/" << nrxx << std::endl; 
    }
    int ifunc_index = 0;

    for ( xc_func_type &func : funcs )
    {
        //std::cout << "processing functional " << ifunc_index << std::endl;
        ifunc_index += 1;
        assert(func.info->family == XC_FAMILY_MGGA);
        xc_mgga_exc_vxc(&func, nrxx, rho.data(), sigma.data(), sigma.data(),
            kin_r.data(), exc.data(), vrho.data(), vsigma.data(), vlapl.data(), vtau.data());

        // rho[ir*2], rho[ir*2+1], sigma[ir*3], sigma[ir*3+2], tau[ir*2], tau[ir*2+1], sgn[ir*2], sgn[ir*2+1],
        // exc[ir], vrho[ir*2], vrho[ir*2+1], vsigma[ir*3], vsigma[ir*3+2], vtau[ir*2], vtau[ir*2+1]
        // for ir_max_discard_rhoa, ir_max_discard_rhob, ir_max_discard_sigma_aa, ir_max_discard_sigma_bb, ir_max_discard_tau_a, ir_max_discard_tau_b
        // using scientific notation, with n_digits digits after the decimal point, and total length of n_len
        // n_digits = 2, n_len = 10
        int n_digits = 2;
        int n_len = 10;
        std::cout << "rho, sigma, tau, sgn, exc, vrho, vsigma, vtau for the maximum discarded alpha and beta points for functional index: " << ifunc_index  << std::endl;
        std::cout << std::scientific << std::setprecision(n_digits);
        /// do a loop over ir for ir in {ir_max_discard_rhoa, ir_max_discard_rhob, ir_max_discard_sigma_aa, ir_max_discard_sigma_bb, ir_max_discard_tau_a, ir_max_discard_tau_b}
        std::cout << std::setw(n_len) << "ir"
                  << std::setw(n_len) << "rho_a"
                  << std::setw(n_len) << "rho_b"
                  << std::setw(n_len) << "sigma_aa"
                  << std::setw(n_len) << "sigma_bb"
                  << std::setw(n_len) << "tau_a"
                  << std::setw(n_len) << "tau_b"
                  << std::setw(n_len) << "sgn_a"
                  << std::setw(n_len) << "sgn_b"
                  << std::setw(n_len) << "exc"
                  << std::setw(n_len) << "vrho_a"
                  << std::setw(n_len) << "vrho_b"
                  << std::setw(n_len) << "vsigma_aa"
                  << std::setw(n_len) << "vsigma_bb"
                  << std::setw(n_len) << "vtau_a"
                  << std::setw(n_len) << "vtau_b" 
                  << std::endl;
        
            std::list<int> ir_list = {
                ir_max_discard_rhoa,
                ir_max_discard_rhob,
                ir_max_discard_sigma_aa,
                ir_max_discard_sigma_bb,
                ir_max_discard_tau_a,
                ir_max_discard_tau_b
            };

        for (int ir : ir_list)
        {
            std::cout << std::setw(n_len) << ir
                      << std::setw(n_len) << rho[ir*2]
                      << std::setw(n_len) << rho[ir*2+1]
                      << std::setw(n_len) << sigma[ir*3]
                      << std::setw(n_len) << sigma[ir*3+2]
                      << std::setw(n_len) << kin_r[ir*2]
                      << std::setw(n_len) << kin_r[ir*2+1]
                      << std::setw(n_len) << sgn[ir*2]
                      << std::setw(n_len) << sgn[ir*2+1]
                      << std::setw(n_len) << exc[ir]
                      << std::setw(n_len) << vrho[ir*2]
                      << std::setw(n_len) << vrho[ir*2+1]
                      << std::setw(n_len) << vsigma[ir*3]
                      << std::setw(n_len) << vsigma[ir*3+2]
                      << std::setw(n_len) << vtau[ir*2]
                      << std::setw(n_len) << vtau[ir*2+1] 
                      << std::endl;
        }


        std::cout << std::endl;
        std::cout << std::endl;
        


        //process etxc
        for( int is=0; is!=nspin; ++is )
        {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) schedule(static, 256)
#endif
            for( int ir=0; ir< nrxx; ++ir )
            {
#ifdef __EXX
                if (func.info->number == XC_MGGA_X_SCAN && XC_Functional::get_func_type() == 5)
                {
                    exc[ir] *= (1.0 - XC_Functional::get_hybrid_alpha());
                }
#endif
                etxc += ModuleBase::e2 * exc[ir] * rho[ir*nspin+is]  * sgn[ir*nspin+is];
            }
            std::cout << "etxc after all grid points: " << etxc << std::endl;

        }

        //process vtxc
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:vtxc) schedule(static, 256)
#endif
        for( int is=0; is<nspin; ++is )
        {
            for( int ir=0; ir< nrxx; ++ir )
            {
#ifdef __EXX
                if (func.info->number == XC_MGGA_X_SCAN && XC_Functional::get_func_type() == 5)
                {
                    vrho[ir*nspin+is] *= (1.0 - XC_Functional::get_hybrid_alpha());
                }
#endif
                const double v_tmp = ModuleBase::e2 * vrho[ir*nspin+is]  * sgn[ir*nspin+is];
//                v(is,ir) += v_tmp * rho[ir*nspin+is]; // issue 1
                v(is,ir) += v_tmp;
                vtxc += v_tmp * chr->rho[is][ir];
            }
        }
        std::cout << "vtxc after all grid points: " << vtxc << std::endl;

        //process vsigma
        std::vector<std::vector<ModuleBase::Vector3<double>>> h(
            nspin,
            std::vector<ModuleBase::Vector3<double>>(nrxx) );
        if( 1==nspin )
        {
            double sum_abs_h = 0.0;
            double sum_abs_gdr = 0.0;
            double sum_abs_vsigma = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for( int ir=0; ir< nrxx; ++ir )
            {
#ifdef __EXX
                if (func.info->number == XC_MGGA_X_SCAN && XC_Functional::get_func_type() == 5)
                {
                    vsigma[ir] *= (1.0 - XC_Functional::get_hybrid_alpha());
                }
#endif
                h[0][ir] = 2.0 * gdr[0][ir] * vsigma[ir] * 2.0 * sgn[ir];
                if (std::abs(vsigma[ir]) > 1.0e3)
                {
                    std::cout << rho[ir]    << " " << 
                                 sigma[ir]  << " " << 
                                 kin_r[ir]  << " " <<
                                 sgn[ir]    << " " <<
                                 exc[ir]    << " " <<
                                 vrho[ir]   << " " <<
                                 vsigma[ir] << " " <<
                                 vtau[ir]   << " " << std::endl;
                }
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 64)
#endif
            for( int ir=0; ir< nrxx; ++ir )
            {
#ifdef __EXX
                if (func.info->number == XC_MGGA_X_SCAN && XC_Functional::get_func_type() == 5)
                {
                    vsigma[ir*3]   *= (1.0 - XC_Functional::get_hybrid_alpha());
                    vsigma[ir*3+1] *= (1.0 - XC_Functional::get_hybrid_alpha());
                    vsigma[ir*3+2] *= (1.0 - XC_Functional::get_hybrid_alpha());
                }
#endif
                h[0][ir] = 2.0 * (gdr[0][ir] * vsigma[ir*3  ] * sgn[ir*2  ] * 2.0
                                + gdr[1][ir] * vsigma[ir*3+1] * sgn[ir*2]   * sgn[ir*2+1]);
                h[1][ir] = 2.0 * (gdr[1][ir] * vsigma[ir*3+2] * sgn[ir*2+1] * 2.0
                                + gdr[0][ir] * vsigma[ir*3+1] * sgn[ir*2]   * sgn[ir*2+1]);
            }
        }

        // define two dimensional array dh [ nspin, nrxx ]
        std::vector<std::vector<double>> dh(nspin, std::vector<double>( nrxx));
        for( int is=0; is!=nspin; ++is )
        {
            XC_Functional::grad_dot( h[is].data(),
                dh[is].data(), chr->rhopw,
                tpiba);
            // compute dh[0] as 
        }
        double sum_abs_dh =0.0;
        if (nspin == 1) {
            for( int ir=0; ir< nrxx; ++ir ){
                sum_abs_dh += std::abs(dh[0][ir]);
            }
            std::cout << "sum of abs(dh) after all grid points: " << sum_abs_dh << std::endl;
        }

        double rvtxc = 0.0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:rvtxc) schedule(static, 256)
#endif
        for( int is=0; is<nspin; ++is )
        {
            //std::cout << "do I reach here 3? " << is << std::endl;
            for( int ir=0; ir< nrxx; ++ir )
            {
                rvtxc += dh[is][ir] * rho[ir*nspin+is];
                v(is,ir) -= dh[is][ir];
            }
        }
        vtxc -= rvtxc;
        /// print dh for first 5 grid points
        /// using scientific notation, with n_digits digits after the decimal point, and total length of n_len
        /// n_digits = 2, n_len = 10
        //int n_digits = 2;
        //int n_len = 10;
        std::cout << "h and dh for first 5 grid points in xc_functional_libxc:" << std::endl;
        std::cout << std::scientific << std::setprecision(n_digits);
        std::cout << std::setw(n_len) << "ir"
                  << std::setw(n_len) << "h_a"
                  << std::setw(n_len) << "h_b"   
                  << std::setw(n_len) << "dh_a"
                  << std::setw(n_len) << "dh_b" << std::endl;
        for (int ir = 0; ir < std::min(n_grid_print, nrxx); ir++)
        {
            std::cout << std::setw(n_len) << ir
                      << std::setw(n_len) << h[0][ir]
                      << std::setw(n_len) << h[1][ir]
                      << std::setw(n_len) << dh[0][ir]
                      << std::setw(n_len) << dh[1][ir] << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
        float mod_ha = 0.0;
        float mod_hb = 0.0;
        // loop over all ir < nrxx, calculate sqrt(h[0][ir].x^2 + h[0][ir].y^2 + h[0][ir].z^2).
        // if it is greater than 1.0e4, print the values of rho, sigma, kin_r, sgn, exc, vrho, vsigma, vtau for ir.

        // loop over all ir < nrxx, find the maximum absolute value for exc[ir], vrho[ir*2] as vrho_a, vrho[ir*2+1] as vrho_b,
        // sigma[ir*3] as vsigma_aa, sigma[ir*3+2] as vsigma_bb,
        // kin_r[ir*2] as vtau_a, kin_r[ir*2+1] as vtau_b,
        // when the loop is done, print rho[ir*2], rho[ir*2+1], sigma[ir*3], sigma[ir*3+2],
        // kin_r[ir*2], kin_r[ir*2+1], sgn[ir*2], sgn[ir*2+1],
        // exc[ir], vrho[ir*2], vrho[ir*2+1], vsigma[ir*3], vsigma[ir*3+2],
        // vtau[ir*2], vtau[ir*2+1] for the ir with each of the maximum values
        int max_ir_exc = 0;
        int max_ir_vrho_a = 0;
        int max_ir_vrho_b = 0;
        int max_ir_vsigma_aa = 0;
        int max_ir_vsigma_bb = 0;
        int max_ir_vtau_a = 0;
        int max_ir_vtau_b = 0;
        double max_exc = 0.0;
        double max_vrho_a = 0.0;
        double max_vrho_b = 0.0;
        double max_vsigma_aa = 0.0;
        double max_vsigma_bb = 0.0;
        double max_vtau_a = 0.0;
        double max_vtau_b = 0.0;

        for (int ir = 0; ir < nrxx; ir++)
        {
            if (std::abs(exc[ir]) > max_exc && sgn[ir*nspin] > 0.0 && sgn[ir*nspin+1] > 0.0)
            {
                max_ir_exc = ir;
                max_exc = std::abs(exc[ir]);
            }
            if (std::abs(vrho[ir*nspin]) > max_vrho_a && sgn[ir*nspin] > 0.0)
            {
                max_ir_vrho_a = ir;
                max_vrho_a = std::abs(vrho[ir*nspin]);
            }
            if (std::abs(vrho[ir*nspin+1]) > max_vrho_b && sgn[ir*nspin+1] > 0.0)
            {
                max_ir_vrho_b = ir;
                max_vrho_b = std::abs(vrho[ir*nspin+1]);
            }
            if (std::abs(vsigma[ir*3]) > max_vsigma_aa && sgn[ir*2] > 0.0)
            {
                max_ir_vsigma_aa = ir;
                max_vsigma_aa = std::abs(vsigma[ir*3]);
            }
            if (std::abs(vsigma[ir*3+2]) > max_vsigma_bb && sgn[ir*2+1] > 0.0)
            {
                max_ir_vsigma_bb = ir;
                max_vsigma_bb = std::abs(vsigma[ir*3+2]);
            }
            if (std::abs(vtau[ir*nspin]) > max_vtau_a && sgn[ir*nspin] > 0.0)
            {
                max_ir_vtau_a = ir;
                max_vtau_a = std::abs(vtau[ir*nspin]);
            }
            if (std::abs(vtau[ir*nspin+1]) > max_vtau_b && sgn[ir*nspin+1] > 0.0)
            {
                max_ir_vtau_b = ir;
                max_vtau_b = std::abs(vtau[ir*nspin+1]);
            }
        }

        std::cout << "maximum absolute value of exc, vrho_a, vrho_b, vsigma_aa, vsigma_bb, vtau_a and vtau_b in xc_functional_libxc:" << std::endl;
        std::cout << std::setw(n_len) << "ir"
                  << std::setw(n_len) << "rho_a"
                  << std::setw(n_len) << "rho_b"
                  << std::setw(n_len) << "sigma_aa"
                  << std::setw(n_len) << "sigma_bb"
                  << std::setw(n_len) << "tau_a"
                  << std::setw(n_len) << "tau_b"
                  << std::setw(n_len) << "sgn_a"
                  << std::setw(n_len) << "sgn_b"
                  << std::setw(n_len) << "exc"
                  << std::setw(n_len) << "vrho_a"
                  << std::setw(n_len) << "vrho_b"
                  << std::setw(n_len) << "vsigma_aa"
                  << std::setw(n_len) << "vsigma_bb"
                  << std::setw(n_len) << "vtau_a"
                  << std::setw(n_len) << "vtau_b" << std::endl;
        ir_list = {
            max_ir_exc,
            max_ir_vrho_a,
            max_ir_vrho_b,
            max_ir_vsigma_aa,
            max_ir_vsigma_bb,
            max_ir_vtau_a,
            max_ir_vtau_b
        };
        for (int ir : ir_list)
        {
            std::cout << std::setw(n_len) << ir
                      << std::setw(n_len) << rho[ir*2]
                      << std::setw(n_len) << rho[ir*2+1]
                      << std::setw(n_len) << sigma[ir*3]
                      << std::setw(n_len) << sigma[ir*3+2]
                      << std::setw(n_len) << kin_r[ir*2]
                      << std::setw(n_len) << kin_r[ir*2+1]
                      << std::setw(n_len) << sgn[ir*2]
                      << std::setw(n_len) << sgn[ir*2+1]
                      << std::setw(n_len) << exc[ir]
                      << std::setw(n_len) << vrho[ir*2]
                      << std::setw(n_len) << vrho[ir*2+1]
                      << std::setw(n_len) << vsigma[ir*3]
                      << std::setw(n_len) << vsigma[ir*3+2]
                      << std::setw(n_len) << vtau[ir*2]
                      << std::setw(n_len) << vtau[ir*2+1] << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
        

        //process vtau
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
        for( int is=0; is<nspin; ++is )
        {
            for( int ir=0; ir< nrxx; ++ir )
            {
#ifdef __EXX
                if (func.info->number == XC_MGGA_X_SCAN && XC_Functional::get_func_type() == 5)
                {
                    vtau[ir*nspin+is] *= (1.0 - XC_Functional::get_hybrid_alpha());
                }
#endif
                vofk(is,ir) += vtau[ir*nspin+is]  * sgn[ir*nspin+is];
            }
        }
    }

    //-------------------------------------------------
    // for MPI, reduce the exchange-correlation energy
    //-------------------------------------------------
#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif

    etxc *= omega / chr->rhopw->nxyz;
    vtxc *= omega / chr->rhopw->nxyz;

    XC_Functional_Libxc::finish_func(funcs);

    // print i_reach_2 and nrxx
    //std::cout << "i_reach_3: " << i_reach_3 << " " << "nrxx: " << nrxx << std::endl;
 
    ModuleBase::timer::tick("XC_Functional_Libxc","v_xc_meta");
    /// print v and vofk for ir < 5, and is = 1 and 2, and the sgn with a title as follows:
    /// using scientific notation, with n_digits digits after the decimal point, and total length of n_len
    /// n_digits = 1, n_len = 10
    /// ir rho_a rho_b sig_aa sig_bb tau_a tau_b v_a   v_b   vofk_a   vofk_b   sgn_a  sgn_b
    int n_digits = 2;
    int n_len = 10;
    //std::cout << "v and vofk for first 5 grid points in xc_functional_libxc:" << std::endl;
    //std::cout << std::scientific << std::setprecision(n_digits);
    //std::cout << std::setw(3) << "ir"
    //          << std::setw(n_len) << "rho_a"
    //          << std::setw(n_len) << "rho_b"
    //          << std::setw(n_len) << "sigma_a"
    //          << std::setw(n_len) << "sigma_b"
    //          << std::setw(n_len) << "tau_a"
    //          << std::setw(n_len) << "tau_b"
    //          << std::setw(n_len) << "v_a"
    //          << std::setw(n_len) << "v_b"
    //          << std::setw(n_len) << "vofk_a"
    //          << std::setw(n_len) << "vofk_b"
    //          << std::setw(n_len) << "sgn_a"
    //          << std::setw(n_len) << "sgn_b" << std::endl;
    //for (int ir = 0; ir < std::min(n_grid_print, nrxx); ir++)
    //{
    //    std::cout << std::setw(3) << ir
    //            << std::setw(n_len) << rho[ir*nspin]
    //                << std::setw(n_len) << rho[ir*nspin+1]
    //                << std::setw(n_len) << sigma[ir*nspin]
    //                << std::setw(n_len) << sigma[ir*nspin+1]
    //                << std::setw(n_len) << kin_r[ir*nspin]
    //                << std::setw(n_len) << kin_r[ir*nspin+1]
    //                << std::setw(n_len) << v(0,ir)
    //                << std::setw(n_len) << v(1,ir)
    //                << std::setw(n_len) << vofk(0,ir)
    //                << std::setw(n_len) << vofk(1,ir)
    //                << std::setw(n_len) << sgn[ir*nspin]
    //                << std::setw(n_len) << sgn[ir*nspin+1] << std::endl;
    //}
    //std::cout << std::endl;
    //std::cout << std::endl;
    return std::make_tuple( etxc, vtxc, std::move(v), std::move(vofk) );
}

#endif
