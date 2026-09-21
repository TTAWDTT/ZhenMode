MODULE dynadv_up3
   !!======================================================================
   !!                       ***  MODULE  dynadv_up3  ***
   !! Ocean dynamics: Update the momentum trend with the flux form advection
   !!                 trend using a 3rd order upstream biased scheme
   !!======================================================================
   !! History :  2.0  ! 2006-08  (R. Benshila, L. Debreu)  Original code
   !!            3.2  ! 2009-07  (R. Benshila)  Suppression of rigid-lid option
   !!            4.5  ! 2022-06  (S. Techene, G, Madec) refactorization to reduce local memory usage
   !!            5.x  ! 2023-05  (A. Nasser, S. Techene, G. Madec) remove 4th order divergence and 
   !!                                                              implement 3rd order vertical advection
   !!             -   ! 2024-01  (S. Techene, G. Madec) RK3 optimized 2D RHS computation
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dyn_adv_up3   : flux form momentum advection using    (ln_dynadv=T)
   !!                   an 3rd order Upstream Biased Scheme or Quick scheme
   !!                   combined with 2nd or 4th order finite differences 
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and tracers
   USE dom_oce        ! ocean space and time domain
   USE trd_oce        ! trends: ocean variables
   USE trddyn         ! trend manager: dynamics
   !
   USE in_out_manager ! I/O manager
   USE prtctl         ! Print control
   USE lib_mpp        ! MPP library

   IMPLICIT NONE
   PRIVATE
   
   ! theoretical gamma1 value is multiplied by 2 as we factorise 1/2 on the 3rd order biased part
   REAL(wp), PARAMETER :: gamma1 = 1._wp/3._wp  ! =1/4 quick      ; =1/3  3rd order UBS
   REAL(wp), PARAMETER :: gamma2 = 1._wp/32._wp ! =0   2nd order  ; =1/32 4th order centred
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_thres_dynup3 = 1.35_wp  ! starting Courant number threshold for adaptive implicit vertical advection
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_cstra_dynup3 = 1.60_wp  ! stability constraint for up3 with RK3 (=1.626 in Lemarie et al 2015)

   PUBLIC   dyn_adv_up3        ! routine called by stp2d.F90 and dynadv.F90

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE dyn_adv_up3( kt, Kbb, Kmm, puu, pvv, Krhs, pFu, pFv, pFw, pUe, pVe )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dyn_adv_up3  ***
      !!
      !! ** Purpose :   Compute the now momentum advection trend in flux form
      !!              and the general trend of the momentum equation.
      !!
      !! ** Method  :   The scheme is the one implemeted in ROMS. It depends 
      !!      on gamma1 parameter. It controls the upstream baised part of
      !!      the scheme:     gamma1 = 0    pure centered  (no diffusive part)
      !!                       = 1/4  Quick scheme
      !!                       = 1/3  3rd order Upstream biased scheme
      !!      For stability reasons, the first term of the fluxes which cor-
      !!      responds to a second order centered scheme is evaluated using  
      !!      the now velocity (centered in time) while the second term which  
      !!      is the diffusive part of the scheme, is evaluated using the 
      !!      before velocity (forward in time). 
      !!      Default value (hard coded in the begining of the module) is 
      !!      gamma1=1/3.
      !!
      !!                In RK3 time stepping case, the optional arguments 
      !!      (pFu,pFv,pFw) are present. They are used as advective velocity  
      !!      while the advected velocity remains (puu,pvv). 
      !!
      !! ** Action  :   (puu,pvv)_Krhs   updated with the advective trend
      !!        or only (pUe,pVe)   vertically averaged advective trend
      !!
      !! Reference : Shchepetkin & McWilliams, 2005, Ocean Modelling. 
      !!----------------------------------------------------------------------
      INTEGER                                           , INTENT(in   ) ::   kt , Kbb, Kmm, Krhs   ! ocean time-step and level indices
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt)      , TARGET, INTENT(inout) ::   puu, pvv              ! ocean velocities and RHS of momentum equation
      REAL(wp), DIMENSION(jpi,jpj,jpk), OPTIONAL, TARGET, INTENT(in   ) ::   pFu, pFv, pFw         ! advective velocity
      REAL(wp), DIMENSION(A2D(0))     , OPTIONAL        , INTENT(inout) ::   pUe, pVe              ! advective RHS of the external mode (stp2D)
      !
      LOGICAL  ::   lltrp        ! local logical
      INTEGER  ::   ji, jj, jk   ! dummy loop indices
      REAL(wp) ::   zzu, zui, zFuj, zl_u, zzFu_kp1, zlu_uw_kp1, zFwi     ! local scalars
      REAL(wp) ::   zzv, zvj, zFvi, zl_v, zzFv_kp1, zlv_vw_kp1, zFwj     !   -      -
      REAL(wp), DIMENSION(T2D(1)) ::   zFu_t, zFu_f
      REAL(wp), DIMENSION(T2D(1)) ::   zFv_t, zFv_f
      REAL(wp), DIMENSION(T2D(1)) ::   zlu_uu, zlu_uv
      REAL(wp), DIMENSION(T2D(1)) ::   zlv_vv, zlv_vu
      REAL(wp), DIMENSION(:,:)  , POINTER             ::   zFu, zFv
#if ! defined key_PSYCLONE_2p5p0
      REAL(wp), DIMENSION(:,:)  , ALLOCATABLE, TARGET ::   zwu, zwv
#else
      REAL(wp), DIMENSION(T2D(1)), TARGET             ::   zwu, zwv
#endif
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE         ::   zu_trd, zv_trd
      !!----------------------------------------------------------------------
      !
      IF( .NOT. l_istiled .OR. ntile == 1 )  THEN                       ! Do only on the first tile
         IF( kt == nit000 ) THEN
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) 'dyn_adv_up3 : UP3 flux form momentum advection'
            IF(lwp) WRITE(numout,*) '~~~~~~~~~~~'
            lltrp = PRESENT( pFu ) .AND. PRESENT( pFv ) .AND. PRESENT( pFw )
            IF(  .NOT.lltrp .AND. ( PRESENT( pFu ) .OR. PRESENT( pFv ) .OR. PRESENT( pFw ) )  )   &
               &     CALL ctl_stop('STOP','dynadv_up3: put all the 3 arguments (pFu, pFv, and pFw) or none of them' )
            lltrp = PRESENT( pUe ) .AND. PRESENT( pVe )
            IF(  .NOT. lltrp .AND. ( PRESENT( pUe ) .OR. PRESENT( pVe ) )  )   &
               &     CALL ctl_stop('STOP','dynadv_up3: put all the 2 arguments both (pUe, pVe) or none of them' )
         ENDIF
      ENDIF
      !                             ! check the presence of 3D advective transport 
      !
      IF( l_trddyn ) THEN           ! trends: send trend to trddyn for diagnostic  
         ALLOCATE( zu_trd(T2D(0),jpk), zv_trd(T2D(0),jpk) )
         zu_trd(:,:,:) = puu(T2D(0),:,Krhs)
         zv_trd(:,:,:) = pvv(T2D(0),:,Krhs)
      ENDIF
      !                             ! used in stp2d : advective velocity = (puu,pvv,ww)
#if ! defined key_PSYCLONE_2p5p0
      IF( .NOT. PRESENT( pFu ) )   ALLOCATE( zwu(T2D(1)), zwv(T2D(1)) )
#endif
      
      IF( PRESENT( pUe ) ) THEN     ! 3D RHS cumulation : set 2D RHS to zero
         DO_2D( 0, 0, 0, 0 )
            pUe(ji,jj) = 0._wp   ;   pVe(ji,jj) = 0._wp
         END_2D
      ENDIF
      !
      !                             ! =============================== !
      DO jk = 1, jpkm1              !  Horizontal advection : k-slab  !
         !                          ! =============================== !
         !
         !            
         DO_2D( 1, 1, 1, 1 )                       ! laplacian of the horizontal velocity
            zlu_uu(ji,jj) = (  ( puu(ji+1,jj  ,jk,Kbb) - puu(ji  ,jj  ,jk,Kbb) )  &                      ! add () for NP repro
               &             + ( puu(ji-1,jj  ,jk,Kbb) - puu(ji  ,jj  ,jk,Kbb) )  ) * umask(ji  ,jj  ,jk)
            zlv_vv(ji,jj) = (  ( pvv(ji  ,jj+1,jk,Kbb) - pvv(ji  ,jj  ,jk,Kbb) )  &                      ! add () for NP repro
               &             + ( pvv(ji  ,jj-1,jk,Kbb) - pvv(ji  ,jj  ,jk,Kbb) )  ) * vmask(ji  ,jj  ,jk)
            zlu_uv(ji,jj) =    ( puu(ji  ,jj+1,jk,Kbb) - puu(ji  ,jj  ,jk,Kbb) )    * fmask(ji  ,jj  ,jk)   &
               &             - ( puu(ji  ,jj  ,jk,Kbb) - puu(ji  ,jj-1,jk,Kbb) )    * fmask(ji  ,jj-1,jk)
            zlv_vu(ji,jj) =    ( pvv(ji+1,jj  ,jk,Kbb) - pvv(ji  ,jj  ,jk,Kbb) )    * fmask(ji  ,jj  ,jk)   &
               &             - ( pvv(ji  ,jj  ,jk,Kbb) - pvv(ji-1,jj  ,jk,Kbb) )    * fmask(ji-1,jj  ,jk)
         END_2D
         !
         !                                         ! horizontal volume fluxes
         IF( PRESENT( pFu ) ) THEN
            zFu => pFu(:,:,jk)
            zFv => pFv(:,:,jk)
         ELSE
            zFu(T2D(1)) => zwu(:,:)
            zFv(T2D(1)) => zwv(:,:)
            DO_2D( 1, 1, 1, 1 )
               zFu(ji,jj) = e2u(ji,jj) * e3u(ji,jj,jk,Kmm) * puu(ji,jj,jk,Kmm)
               zFv(ji,jj) = e1v(ji,jj) * e3v(ji,jj,jk,Kmm) * pvv(ji,jj,jk,Kmm)
            END_2D
         ENDIF
         !
         DO_2D( 1, 0, 1, 0 )                       ! 4x horizontal momentum fluxes at T- and F-point
            zui = ( puu(ji,jj,jk,Kmm) + puu(ji+1,jj  ,jk,Kmm) )
            zvj = ( pvv(ji,jj,jk,Kmm) + pvv(ji  ,jj+1,jk,Kmm) )
            !
            IF( zui > 0 ) THEN   ;   zl_u = zlu_uu(ji  ,jj)
            ELSE                 ;   zl_u = zlu_uu(ji+1,jj)
            ENDIF
            IF( zvj > 0 ) THEN   ;   zl_v = zlv_vv(ji,jj  )
            ELSE                 ;   zl_v = zlv_vv(ji,jj+1)
            ENDIF
            !
            zFu_t(ji+1,jj  ) = (  zFu(ji,jj) + zFu(ji+1,jj  )  ) * ( zui - gamma1 * zl_u )
            zFv_t(ji  ,jj+1) = (  zFv(ji,jj) + zFv(ji  ,jj+1)  ) * ( zvj - gamma1 * zl_v )
            !
            zFuj = ( zFu(ji,jj) + zFu(ji  ,jj+1) )
            zFvi = ( zFv(ji,jj) + zFv(ji+1,jj  ) )
            !
            IF( zFuj > 0 ) THEN   ;    zl_v = zlv_vu(ji  ,jj)
            ELSE                  ;    zl_v = zlv_vu(ji+1,jj)
            ENDIF
            IF( zFvi > 0 ) THEN   ;    zl_u = zlu_uv(ji,jj  )
            ELSE                  ;    zl_u = zlu_uv(ji,jj+1)
            ENDIF
            !
            zFv_f(ji  ,jj  ) = zFvi * ( ( puu(ji,jj,jk,Kmm) + puu(ji  ,jj+1,jk,Kmm) ) - gamma1 * zl_u )   ! add () for NP repro
            zFu_f(ji  ,jj  ) = zFuj * ( ( pvv(ji,jj,jk,Kmm) + pvv(ji+1,jj  ,jk,Kmm) ) - gamma1 * zl_v )   ! add () for NP repro
         END_2D
         !
         IF( PRESENT( pUe ) ) THEN      !-  2D RHS : vertically cumulated  -!   (here MASKED jk-sum)
!!gm TO BE tested : UNmasked horizontal trends and NO vertical component   !!!
            DO_2D( 0, 0, 0, 0 )                       ! divergence of horizontal momentum fluxes
               pUe(ji,jj) = pUe(ji,jj) - 0.25_wp * (  ( zFu_t(ji+1,jj) - zFu_t(ji,jj  ) )    &   ! add () for NP repro
                  &                                 + ( zFv_f(ji  ,jj) - zFv_f(ji,jj-1) )  ) * r1_e1e2u(ji,jj)   &
                  &                              * umask(ji,jj,jk)
               pVe(ji,jj) = pVe(ji,jj) - 0.25_wp * (  ( zFu_f(ji,jj  ) - zFu_f(ji-1,jj) )    &   ! add () for NP repro
                  &                                 + ( zFv_t(ji,jj+1) - zFv_t(ji  ,jj) )  ) * r1_e1e2v(ji,jj)   &
                  &                              * vmask(ji,jj,jk)
            END_2D
         ELSE                           !-  added the 3D RHS  -!
            DO_2D( 0, 0, 0, 0 )                       ! divergence of horizontal momentum fluxes
               puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp * (  ( zFu_t(ji+1,jj) - zFu_t(ji,jj  ) )    &   ! add () for NP repro
                  &                                                 + ( zFv_f(ji  ,jj) - zFv_f(ji,jj-1) )  ) * r1_e1e2u(ji,jj)   &
                  &                                    / e3u(ji,jj,jk,Kmm)
               pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - 0.25_wp * (  ( zFu_f(ji,jj  ) - zFu_f(ji-1,jj) )    &   ! add () for NP repro
                  &                                                 + ( zFv_t(ji,jj+1) - zFv_t(ji  ,jj) )  ) * r1_e1e2v(ji,jj)   &
                  &                                    / e3v(ji,jj,jk,Kmm)
            END_2D
         ENDIF
         !                          ! =============== !
      END DO                        !  End of k-slab  !
      !                             ! =============== !
      !
      IF( l_trddyn ) THEN           ! trends: send trend to trddyn for diagnostic
         ALLOCATE( zu_trd(A2D(2),jpkm1), zv_trd(A2D(2),jpkm1) )
         zu_trd(A2D(0),:) = puu(A2D(0),1:jpkm1,Krhs)
         zv_trd(A2D(0),:) = pvv(A2D(0),1:jpkm1,Krhs)
      ENDIF

!!gm TO BE tested : UNmasked horizontal trends and NO vertical component   !!!
!!     in this case averaging done here + return
!      IF( PRESENT( pUe ) ) THEN   ! averraging
!         DO_2D( 0, 0, 0, 0 )
!            puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) / hu(ji,jj,Kmm)
!            pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) / hv(ji,jj,Kmm)
!         END DO
!         !
!         RETURN     ! NO vertical advection component (except in linear ssh  ==>>> to be added before the return or in step2D) 
!      ENDIF
!!gm




      !                              ! ========================== !
      !                              !  Vertical advection k-slab !
      !                              ! ========================== !
#define zFwu     zFu_t
#define zFwv     zFv_t
#define zFw      zFu
#define zww      zwu
#define zlu_uw   zFu_f
#define zlv_vw   zFv_f
      !                                   != surface vertical fluxes =! (jk = 1)
      !
      IF( lk_linssh ) THEN                       ! linear free surface: advection through the surface z=0
         IF( PRESENT( pFu ) ) THEN
            zFw => pFw(:,:,1)
         ELSE
            zFw(T2D(1)) => zww(:,:)
            DO_2D( 0, 1, 0, 1 )
               zFw(ji,jj) = e1e2t(ji,jj) * ww(ji,jj,1)
            END_2D
         ENDIF
         !
         DO_2D( 0, 0, 0, 0 )
            zFwu(ji,jj) = 0.5_wp * ( zFw(ji,jj) + zFw(ji+1,jj  ) ) * puu(ji,jj,1,Kmm)
            zFwv(ji,jj) = 0.5_wp * ( zFw(ji,jj) + zFw(ji  ,jj+1) ) * pvv(ji,jj,1,Kmm)
         END_2D
      ELSE                                       ! non linear free: surface advective fluxes set to zero
         DO_2D( 0, 0, 0, 0 )
            zFwu(ji,jj) = 0._wp
            zFwv(ji,jj) = 0._wp
         END_2D
      ENDIF
      !
      DO_2D( 0, 0, 0, 0 )                        ! uniform shear in the 1st layer : dz(u(k=1)) = dz(u(k=2)) ==> zlu = 0
         zlu_uw(ji,jj) = 0._wp
         zlv_vw(ji,jj) = 0._wp
      END_2D
      !   
      DO jk = 1, jpk-2                    != divergence of advective fluxes =! (jk = 1 to jpk-2)
         !
         IF( PRESENT( pFu ) ) THEN
            zFw => pFw(:,:,jk+1)
         ELSE
            zFw(T2D(1)) => zww(:,:)
            DO_2D( 0, 1, 0, 1 )                  ! vertical transport at level k+1
               zFw(ji,jj) = e1e2t(ji,jj) * ww(ji,jj,jk+1)
            END_2D
         ENDIF
         !
         IF( PRESENT( pUe ) ) THEN   !-  2D RHS : MASKED vertically cumulated  -!
!!gm TO BE tested : UNmasked horizontal trends and NO vertical component   !!!
            DO_2D( 0, 0, 0, 0 )
               zlu_uw_kp1 = ( puu(ji,jj,jk  ,Kbb) - puu(ji,jj,jk+1,Kbb) ) * wumask(ji,jj,jk+1)   &
                  &       - ( puu(ji,jj,jk+1,Kbb) - puu(ji,jj,jk+2,Kbb) ) * wumask(ji,jj,jk+2)
               zlv_vw_kp1 = ( pvv(ji,jj,jk  ,Kbb) - pvv(ji,jj,jk+1,Kbb) ) * wvmask(ji,jj,jk+1)   &
                  &       - ( pvv(ji,jj,jk+1,Kbb) - pvv(ji,jj,jk+2,Kbb) ) * wvmask(ji,jj,jk+2)
               !
               zFwi = zFw(ji,jj) + zFw(ji+1,jj)
               IF( zFwi > 0 ) THEN   ;   zl_u = zlu_uw_kp1
               ELSE                  ;   zl_u = zlu_uw(ji,jj)
               ENDIF
               zFwj = zFw(ji,jj) + zFw(ji,jj+1)
               IF( zFwj > 0 ) THEN   ;   zl_v = zlv_vw_kp1
               ELSE                  ;   zl_v = zlv_vw(ji,jj)
               ENDIF
               !                                    ! vertical flux at level k+1
               zzFu_kp1 = 0.25_wp * ( zFw(ji,jj) + zFw(ji+1,jj  ) ) * ( puu(ji,jj,jk+1,Kmm) + puu(ji,jj,jk,Kmm) - gamma1 * zl_u )
               zzFv_kp1 = 0.25_wp * ( zFw(ji,jj) + zFw(ji  ,jj+1) ) * ( pvv(ji,jj,jk+1,Kmm) + pvv(ji,jj,jk,Kmm) - gamma1 * zl_v )
               !                                    ! divergence of vertical momentum flux
               pUe(ji,jj) = pUe(ji,jj) - ( zFwu(ji,jj) - zzFu_kp1 ) * r1_e1e2u(ji,jj) * umask(ji,jj,jk)
               pVe(ji,jj) = pVe(ji,jj) - ( zFwv(ji,jj) - zzFv_kp1 ) * r1_e1e2v(ji,jj) * vmask(ji,jj,jk) 
               !                                    ! store vertical flux for next level calculation
               zFwu(ji,jj) = zzFu_kp1
               zFwv(ji,jj) = zzFv_kp1
               !
               zlu_uw(ji,jj) = zlu_uw_kp1
               zlv_vw(ji,jj) = zlv_vw_kp1
            END_2D
         ELSE                           !-  added the 3D RHS  -!
            DO_2D( 0, 0, 0, 0 )
               zlu_uw_kp1 = ( puu(ji,jj,jk  ,Kbb) - puu(ji,jj,jk+1,Kbb) ) * wumask(ji,jj,jk+1)   &
                  &       - ( puu(ji,jj,jk+1,Kbb) - puu(ji,jj,jk+2,Kbb) ) * wumask(ji,jj,jk+2)
               zlv_vw_kp1 = ( pvv(ji,jj,jk  ,Kbb) - pvv(ji,jj,jk+1,Kbb) ) * wvmask(ji,jj,jk+1)   &
                  &       - ( pvv(ji,jj,jk+1,Kbb) - pvv(ji,jj,jk+2,Kbb) ) * wvmask(ji,jj,jk+2)
               !
               zFwi = zFw(ji,jj) + zFw(ji+1,jj)
               IF( zFwi > 0 ) THEN   ;   zl_u = zlu_uw_kp1
               ELSE                  ;   zl_u = zlu_uw(ji,jj)
               ENDIF
               zFwj = zFw(ji,jj) + zFw(ji,jj+1)
               IF( zFwj > 0 ) THEN   ;   zl_v = zlv_vw_kp1
               ELSE                  ;   zl_v = zlv_vw(ji,jj)
               ENDIF
               !                                    ! vertical flux at level k+1
               zzFu_kp1 = 0.25_wp * ( zFw(ji,jj) + zFw(ji+1,jj  ) ) * ( puu(ji,jj,jk+1,Kmm) + puu(ji,jj,jk,Kmm) - gamma1 * zl_u )
               zzFv_kp1 = 0.25_wp * ( zFw(ji,jj) + zFw(ji  ,jj+1) ) * ( pvv(ji,jj,jk+1,Kmm) + pvv(ji,jj,jk,Kmm) - gamma1 * zl_v )
               !                                    ! divergence of vertical momentum flux
               puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zFwu(ji,jj) - zzFu_kp1 ) * r1_e1e2u(ji,jj) / e3u(ji,jj,jk,Kmm)
               pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zFwv(ji,jj) - zzFv_kp1 ) * r1_e1e2v(ji,jj) / e3v(ji,jj,jk,Kmm) 
               !                                    ! store vertical flux for next level calculation
               zFwu(ji,jj) = zzFu_kp1
               zFwv(ji,jj) = zzFv_kp1
               !
               zlu_uw(ji,jj) = zlu_uw_kp1
               zlv_vw(ji,jj) = zlv_vw_kp1
            END_2D
         ENDIF
      END DO
      !
      jk = jpkm1                      != compute last level =! (zzFu_kp1 = zzFv_kp1 = 0)
      IF( PRESENT( pUe ) ) THEN           !-  2D RHS : MASKED vertically cumulated  -!
!!gm TO BE tested : UNmasked horizontal trends and NO vertical component   !!!
         DO_2D( 0, 0, 0, 0 )
            pUe(ji,jj) = pUe(ji,jj) - zFwu(ji,jj) * r1_e1e2u(ji,jj) * umask(ji,jj,jk)
            pVe(ji,jj) = pVe(ji,jj) - zFwv(ji,jj) * r1_e1e2v(ji,jj) * vmask(ji,jj,jk)
         END_2D
      ELSE                                !-  added the 3D RHS  -!
         DO_2D( 0, 0, 0, 0 )
            puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - zFwu(ji,jj) * r1_e1e2u(ji,jj) / e3u(ji,jj,jk,Kmm)
            pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - zFwv(ji,jj) * r1_e1e2v(ji,jj) / e3v(ji,jj,jk,Kmm) 
         END_2D
      ENDIF
      !
      IF( PRESENT( pUe ) ) THEN       != averaging =!
         DO_2D( 0, 0, 0, 0 )
            pUe(ji,jj) = pUe(ji,jj) * r1_hu(ji,jj,Kmm)
            pVe(ji,jj) = pVe(ji,jj) * r1_hv(ji,jj,Kmm)
         END_2D
      ENDIF
      !
      IF( l_trddyn ) THEN                 ! trends: send trend to trddyn for diagnostic
         zu_trd(A2D(0),:) = puu(A2D(0),1:jpkm1,Krhs) - zu_trd(A2D(0),:)
         zv_trd(A2D(0),:) = pvv(A2D(0),1:jpkm1,Krhs) - zv_trd(A2D(0),:)
         CALL trd_dyn( zu_trd, zv_trd, jpdyn_zad, kt, Kmm )
         DEALLOCATE( zu_trd, zv_trd )
      ENDIF
      !
#if ! defined key_PSYCLONE_2p5p0
      IF( .NOT. PRESENT( pFu ) )   DEALLOCATE( zwu, zwv ) 
#endif
      !
#undef zFwu
#undef zFwv
#undef zFw
#undef zww
#undef zlu_uw
#undef zlv_vw
      !                                   ! Control print
      IF(sn_cfctl%l_prtctl)   CALL prt_ctl( tab3d_1=puu(:,:,:,Krhs), clinfo1=' up3  adv - Ua: ', mask1=umask,   &
         &                                  tab3d_2=pvv(:,:,:,Krhs), clinfo2=           ' Va: ', mask2=vmask, clinfo3='dyn' )
      !
   END SUBROUTINE dyn_adv_up3

   !!==============================================================================
END MODULE dynadv_up3
