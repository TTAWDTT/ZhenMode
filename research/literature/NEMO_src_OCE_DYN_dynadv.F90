MODULE dynadv
   !!==============================================================================
   !!                       ***  MODULE  dynadv  ***
   !! Ocean dynamics:  advection scheme control
   !!==============================================================================
   !! History :  1.0  !  2006-11  (G. Madec)  Original code
   !!            3.3  !  2010-10  (C. Ethe, G. Madec)  reorganisation of initialisation phase
   !!            3.6  !  2015-05  (N. Ducousso, G. Madec)  add Hollingsworth scheme as an option 
   !!            4.0  !  2017-07  (G. Madec)  add a linear dynamics option
   !!            4.5  !  2022-06  (S. Techene, G, Madec) refactorization to reduce local memory usage
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dyn_adv       : compute the momentum advection trend 
   !!   dyn_adv_init  : control the different options of advection scheme
   !!----------------------------------------------------------------------
   USE dom_oce        ! ocean space and time domain
   USE dynadv_cen2    ! centred flux form advection      (dyn_adv_cen2 routine)
   USE dynadv_up3     ! UP3 flux form advection          (dyn_adv_up3  routine)
   USE dynkeg         ! kinetic energy gradient          (dyn_keg      routine)
   USE dynzad         ! vertical advection               (dyn_zad      routine)
   USE zdf_oce,  ONLY : ln_zad_Aimp
   !
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! MPP library
   USE timing         ! Timing

   IMPLICIT NONE
   PRIVATE

   PUBLIC dyn_adv       ! routine called by stprk3_stg.F90
   PUBLIC dyn_adv_init  ! routine called by nemogcm.F90
 
   !                                   !!* namdyn_adv namelist *
   LOGICAL, PUBLIC ::   ln_dynadv_OFF   !: linear dynamics (no momentum advection)
   LOGICAL, PUBLIC ::   ln_dynadv_vec   !: vector form
   INTEGER, PUBLIC ::      nn_dynkeg       !: scheme of grad(KE): =0 C2 ; =1 Hollingsworth
   LOGICAL, PUBLIC ::   ln_dynadv_cen2  !: flux form - 2nd order centered scheme flag
   LOGICAL, PUBLIC ::   ln_dynadv_up3   !: flux form - 3rd order upstream scheme flag

   INTEGER, PUBLIC ::   n_dynadv   !: choice of the formulation and scheme for momentum advection
   !                               !  associated indices:
   INTEGER,  PUBLIC, PARAMETER ::   np_LIN_dyn = 0   ! no advection: linear dynamics
   INTEGER,  PUBLIC, PARAMETER ::   np_VEC_c2  = 1   ! vector form : 2nd order centered scheme
   INTEGER,  PUBLIC, PARAMETER ::   np_FLX_c2  = 2   ! flux   form : 2nd order centered scheme
   INTEGER,  PUBLIC, PARAMETER ::   np_FLX_up3 = 3   ! flux   form : 3rd order Upstream Biased Scheme
   REAL(wp), PUBLIC            ::   r_stb_thres_dyn_v  ! starting Courant number threshold for adaptive implicit vertical advection
   REAL(wp), PUBLIC            ::   r_stb_cstra_dyn_v  ! stability constraint for dynamic advection (vertical)
   REAL(wp), PUBLIC            ::   r_stb_cstra_dyn_h  ! stability constraint for dynamic advection (horizontal)

   !! * Substitutions
#  include "read_nml_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE dyn_adv( kt, Kbb, Kmm, puu, pvv, Krhs, pau, pav, paw )
      !!---------------------------------------------------------------------
      !!                  ***  ROUTINE dyn_adv  ***
      !!                
      !! ** Purpose :   compute the ocean momentum advection trend.
      !!
      !! ** Method  : - Update (puu(:,:,:,Krhs),pvv(:,:,:,Krhs)) with the advection term following n_dynadv
      !!
      !!      NB: in flux form advection (ln_dynadv_cen2 or ln_dynadv_up3=T) 
      !!      a metric term is add to the coriolis term while in vector form 
      !!      it is the relative vorticity which is added to coriolis term
      !!      (see dynvor module).
      !!----------------------------------------------------------------------
      INTEGER                                          , INTENT(in   ) ::   kt , Kbb, Kmm, Krhs   ! ocean time step and level indices
      REAL(wp), DIMENSION(jpi,jpj,jpk),OPTIONAL, TARGET, INTENT(in   ) ::   pau, pav, paw         ! advective velocity
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt),      TARGET, INTENT(inout) ::   puu, pvv              ! ocean velocities and RHS of momentum Eq.
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start( 'dyn_adv' )
      !
      SELECT CASE( n_dynadv )    !==  compute advection trend and add it to general trend  ==!
      CASE( np_VEC_c2  )                                                         != vector form =!
         !                                                                             !* horizontal gradient of kinetic energy
         CALL dyn_keg     ( kt, nn_dynkeg     , Kmm, puu, pvv, Krhs )
         CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )                  !* vertical advection
         !
      CASE( np_FLX_c2  )                                                         !=  flux form  =!
         CALL dyn_adv_cen2( kt                , Kmm, puu, pvv, Krhs, pau, pav, paw )   !* 2nd order centered scheme
         !
      CASE( np_FLX_up3 )                                                               !* 3rd order upstream scheme (UP3)
         CALL dyn_adv_up3     ( kt       , Kbb, Kmm, puu, pvv, Krhs, pau, pav, paw )
         !
      END SELECT
      !
      IF( ln_timing )   CALL timing_stop( 'dyn_adv' )
      !
   END SUBROUTINE dyn_adv

   
   SUBROUTINE dyn_adv_init
      !!---------------------------------------------------------------------
      !!                  ***  ROUTINE dyn_adv_init  ***
      !!                
      !! ** Purpose :   Control the consistency between namelist options for 
      !!              momentum advection formulation & scheme and set n_dynadv
      !!----------------------------------------------------------------------
      INTEGER ::   ioptio, ios, istat1, istat2   ! Local integer
      !
      NAMELIST/namdyn_adv/ ln_dynadv_OFF, ln_dynadv_vec, nn_dynkeg, ln_dynadv_cen2, ln_dynadv_up3
      !!----------------------------------------------------------------------
      !
      IF(lwp) THEN
         WRITE(numout,*)
         WRITE(numout,*) 'dyn_adv_init : choice/control of the momentum advection scheme'
         WRITE(numout,*) '~~~~~~~~~~~~'
      ENDIF
      !
      READ_NML_REF(numnam,namdyn_adv)
      READ_NML_CFG(numnam,namdyn_adv)
      IF(lwm) WRITE ( numond, namdyn_adv )

      IF(lwp) THEN                    ! Namelist print
         WRITE(numout,*) '   Namelist namdyn_adv : chose a advection formulation & scheme for momentum'
         WRITE(numout,*) '      linear dynamics : no momentum advection          ln_dynadv_OFF  = ', ln_dynadv_OFF
         WRITE(numout,*) '      Vector form: 2nd order centered scheme           ln_dynadv_vec  = ', ln_dynadv_vec
         WRITE(numout,*) '         with Hollingsworth scheme (=1) or not (=0)       nn_dynkeg   = ', nn_dynkeg
         WRITE(numout,*) '      flux form: 2nd order centred scheme              ln_dynadv_cen2 = ', ln_dynadv_cen2
         WRITE(numout,*) '                 3rd order UBS scheme                  ln_dynadv_up3  = ', ln_dynadv_up3
      ENDIF

      ioptio = 0                      ! parameter control and set n_dynadv
      IF( ln_dynadv_OFF  ) THEN   ;   ioptio = ioptio + 1   ;   n_dynadv = np_LIN_dyn   ;   ENDIF
      IF( ln_dynadv_vec  ) THEN   ;   ioptio = ioptio + 1   ;   n_dynadv = np_VEC_c2    ;   r_stb_thres_dyn_v = pp_stb_thres_dync2   ;  r_stb_cstra_dyn_v = pp_stb_cstra_dync2   ;  r_stb_cstra_dyn_h = pp_stb_cstra_dync2  ;  ENDIF
      IF( ln_dynadv_cen2 ) THEN   ;   ioptio = ioptio + 1   ;   n_dynadv = np_FLX_c2    ;   r_stb_thres_dyn_v = pp_stb_thres_dync2   ;  r_stb_cstra_dyn_v = pp_stb_cstra_dync2   ;  r_stb_cstra_dyn_h = pp_stb_cstra_dync2  ;  ENDIF
      IF( ln_dynadv_up3  ) THEN   ;   ioptio = ioptio + 1   ;   n_dynadv = np_FLX_up3   ;   r_stb_thres_dyn_v = pp_stb_thres_dynup3  ;  r_stb_cstra_dyn_v = pp_stb_cstra_dynup3  ;  r_stb_cstra_dyn_h = pp_stb_cstra_dynup3 ;  ENDIF

      IF( ioptio /= 1 )   CALL ctl_stop( 'choose ONE and only ONE advection scheme' )
      IF( nn_dynkeg /= nkeg_C2 .AND. nn_dynkeg /= nkeg_HW )   CALL ctl_stop( 'KEG scheme wrong value of nn_dynkeg' )
#if defined key_qcoTest_FluxForm
      IF( ln_dynadv_vec  ) THEN CALL ctl_stop( 'STOP', 'key_qcoTest_FluxForm requires flux form advection' )
#endif

      IF(lwp) THEN                    ! Print the choice
         WRITE(numout,*)
         SELECT CASE( n_dynadv )
         CASE( np_LIN_dyn )   ;   WRITE(numout,*) '   ==>>>   linear dynamics : no momentum advection used'
         CASE( np_VEC_c2  )   ;   WRITE(numout,*) '   ==>>>   vector form : keg + zad + vor is used' 
            IF( nn_dynkeg == nkeg_C2  )   WRITE(numout,*) '              with Centered standard keg scheme'
            IF( nn_dynkeg == nkeg_HW  )   WRITE(numout,*) '              with Hollingsworth keg scheme'
         CASE( np_FLX_c2  )   ;   WRITE(numout,*) '   ==>>>   flux form   : 2nd order scheme is used'
         CASE( np_FLX_up3 )   ;   WRITE(numout,*) '   ==>>>   flux form   : 3rd order scheme is used'
         END SELECT
      ENDIF
      !
   END SUBROUTINE dyn_adv_init

  !!======================================================================
END MODULE dynadv
