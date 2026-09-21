MODULE dynadv_cen2
   !!======================================================================
   !!                       ***  MODULE  dynadv  ***
   !! Ocean dynamics: Update the momentum trend with the flux form advection
   !!                 using a 2nd order centred scheme
   !!======================================================================
   !! History :  2.0  ! 2006-08  (G. Madec, S. Theetten)  Original code
   !!            3.2  ! 2009-07  (R. Benshila)  Suppression of rigid-lid option
   !!            4.5  ! 2022-06  (S. Techene, G, Madec) refactorization to reduce local memory usage
   !!             -   ! 2024-01  (S. Techene, G. Madec) RK3 optimized 2D RHS computation
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dyn_adv_cen2  : flux form momentum advection (ln_dynadv_cen2=T) using a 2nd order centred scheme  
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and tracers
   USE dom_oce        ! ocean space and time domain
   USE trd_oce        ! trends: ocean variables
   USE trddyn         ! trend manager: dynamics
   !
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! MPP library
   USE prtctl         ! Print control

   IMPLICIT NONE
   PRIVATE

   PUBLIC   dyn_adv_cen2   ! routine called by stp2d.F90 and dynadv.F90
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_thres_dync2 = 1.35_wp  ! starting Courant number threshold for adaptive implicit vertical advection
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_cstra_dync2 = 1.60_wp  ! stability constraint for cen2 with RK3 (=1.73 in Lemarie et al 2015)

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE dyn_adv_cen2( kt, Kmm, puu, pvv, Krhs, pFu, pFv, pFw, pUe, pVe )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dyn_adv_cen2  ***
      !!
      !! ** Purpose :   Compute the momentum advection trend in flux form
      !!              and the general trend of the momentum equation.
      !!
      !! ** Method  :   Trend evaluated with a 2nd order centered scheme 
      !!              using fields at Kmm time-level.
      !!                In RK3 time stepping case, the optional arguments (pFu,pFv,pFw) 
      !!              are present. They are used as advective velocity while 
      !!              the advected velocity remains (puu,pvv). 
      !!
      !! ** Action  :    (puu,pvv)_Krhs    updated with the advective trend
      !!              or only (pUe,PVe) vertically averaged advective trend
      !!----------------------------------------------------------------------
      INTEGER                                           , INTENT(in   ) ::   kt , Kmm, Krhs   ! ocean time-step and level indices
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt)      , TARGET, INTENT(inout) ::   puu, pvv         ! ocean velocities and RHS of momentum equation
      REAL(wp), DIMENSION(jpi,jpj,jpk), OPTIONAL, TARGET, INTENT(in   ) ::   pFu, pFv, pFw    ! advective transport
      REAL(wp), DIMENSION(A2D(0))     , OPTIONAL        , INTENT(inout) ::   pUe, pVe         ! advective RHS of the external mode (stp2D)
      !
      LOGICAL  ::   lltrp            ! local logical 
      INTEGER  ::   ji, jj, jk       ! dummy loop indices
      REAL(wp) ::   zzu, zzFwu_kp1   ! local scalars
      REAL(wp) ::   zzv, zzFwv_kp1   !   -      -
      REAL(wp), DIMENSION(T2D(1)) ::   zFu_t, zFu_f
      REAL(wp), DIMENSION(T2D(1)) ::   zFv_t, zFv_f
      REAL(wp), DIMENSION(:,:)  , POINTER             ::   zFu, zFv, zFw
#if ! defined key_PSYCLONE_2p5p0
      REAL(wp), DIMENSION(:,:)  , ALLOCATABLE, TARGET ::   zwu, zwv
#else
      REAL(wp), DIMENSION(T2D(1)), TARGET             ::   zwu, zwv
#endif
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE         ::   zu_trd, zv_trd
      !!----------------------------------------------------------------------
      !
      IF( kt == nit000 .AND. lwp ) THEN
         IF( .NOT. l_istiled .OR. ntile == 1 )  THEN                       ! Do only on the first tile
            WRITE(numout,*)
            WRITE(numout,*) 'dyn_adv_cen2 : 2nd order flux form momentum advection'
            WRITE(numout,*) '~~~~~~~~~~~~'
         ENDIF
         !                          ! RK3: check the presence of 3D advective transport 
         lltrp = PRESENT( pFu ) .AND. PRESENT( pFv ) .AND. PRESENT( pFw )
         IF( ( .NOT. lltrp ) .AND. ( PRESENT( pFu ) .OR. PRESENT( pFv ) .OR. PRESENT( pFw ) ) )   &
            &     CALL ctl_stop('STOP','dynadv_cen2: provide either 3D or none advective transport (pFu, pFv, pFw)' )
         lltrp = PRESENT( pUe ) .AND. PRESENT( pVe )
         IF(  .NOT. lltrp .AND. ( PRESENT( pUe ) .OR. PRESENT( pVe ) )  )   &
            &     CALL ctl_stop('STOP','dynadv_up3: put all the 2 arguments both (pUe, pVe) or none of them' )
      ENDIF
      !
      IF( l_trddyn ) THEN           ! trends: store the input trends
         ALLOCATE( zu_trd(A2D(2),jpkm1), zv_trd(A2D(2),jpkm1) )
         zu_trd(A2D(0),:) = puu(A2D(0),1:jpkm1,Krhs)
         zv_trd(A2D(0),:) = pvv(A2D(0),1:jpkm1,Krhs)
      ENDIF
      !                             ! used in stp2d : advective velocity = (puu,pvv,ww)
#if ! defined key_PSYCLONE_2p5p0
      IF( .NOT. PRESENT( pFu ) )   ALLOCATE( zwu(T2D(1)), zwv(T2D(1)) )
#endif
      !
      IF( PRESENT( pUe ) ) THEN     ! 3D RHS cumulation : set 2D RHS to zero
         DO_2D( 0, 0, 0, 0 )
            pUe(ji,jj) = 0._wp   ;   pVe(ji,jj) = 0._wp
         END_2D
      ENDIF

      !                             ! =============================== !
      DO jk = 1, jpkm1              !  Horizontal advection : k-slab  !
         !                          ! =============================== !
         !
         IF( PRESENT( pFu ) ) THEN        ! horizontal transport
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
         DO_2D( 1, 0, 1, 0 )              ! horizontal momentum fluxes (at T- and F-point)
            zFu_t(ji+1,jj  ) = ( zFu(ji,jj) + zFu(ji+1,jj) ) * ( puu(ji,jj,jk,Kmm) + puu(ji+1,jj  ,jk,Kmm) )
            zFv_f(ji  ,jj  ) = ( zFv(ji,jj) + zFv(ji+1,jj) ) * ( puu(ji,jj,jk,Kmm) + puu(ji  ,jj+1,jk,Kmm) )
            zFu_f(ji  ,jj  ) = ( zFu(ji,jj) + zFu(ji,jj+1) ) * ( pvv(ji,jj,jk,Kmm) + pvv(ji+1,jj  ,jk,Kmm) )
            zFv_t(ji  ,jj+1) = ( zFv(ji,jj) + zFv(ji,jj+1) ) * ( pvv(ji,jj,jk,Kmm) + pvv(ji  ,jj+1,jk,Kmm) )
         END_2D
         !
         IF( PRESENT( pUe ) ) THEN      !-  2D RHS : vertically cumulated  -!   (here MASKED jk-sum)
            DO_2D( 0, 0, 0, 0 )              ! divergence of horizontal momentum fluxes
               pUe(ji,jj) = pUe(ji,jj) - 0.25_wp * (  ( zFu_t(ji+1,jj) - zFu_t(ji,jj  ) )    &   ! add () for NP repro
                  &                                 + ( zFv_f(ji  ,jj) - zFv_f(ji,jj-1) )  ) * r1_e1e2u(ji,jj) * umask(ji,jj,jk)
               pVe(ji,jj) = pVe(ji,jj) - 0.25_wp * (  ( zFu_f(ji,jj  ) - zFu_f(ji-1,jj) )    &   ! add () for NP repro
                  &                                 + ( zFv_t(ji,jj+1) - zFv_t(ji  ,jj) )  ) * r1_e1e2v(ji,jj) * vmask(ji,jj,jk)
            END_2D
         ELSE                           !-  added the 3D RHS  -!
            DO_2D( 0, 0, 0, 0 )              ! divergence of horizontal momentum fluxes
               puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - (  ( zFu_t(ji+1,jj) - zFu_t(ji,jj  ) )    &   ! add () for NP repro
                  &                                       + ( zFv_f(ji  ,jj) - zFv_f(ji,jj-1) )  ) * 0.25_wp * r1_e1e2u(ji,jj)   &
                  &                                    / e3u(ji,jj,jk,Kmm)
               pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - (  ( zFu_f(ji,jj  ) - zFu_f(ji-1,jj) )    &   ! add () for NP repro
                  &                                       + ( zFv_t(ji,jj+1) - zFv_t(ji  ,jj) )  ) * 0.25_wp * r1_e1e2v(ji,jj)   &
                  &                                    / e3v(ji,jj,jk,Kmm)
            END_2D
         ENDIF
         !                          ! =============== !
      END DO                        !  End of k-slab  !
      !                             ! =============== !
      !
      IF( l_trddyn ) THEN           ! trends: send trend to trddyn for diagnostic
         zu_trd(A2D(0),:) = puu(A2D(0),1:jpkm1,Krhs) - zu_trd(A2D(0),:)
         zv_trd(A2D(0),:) = pvv(A2D(0),1:jpkm1,Krhs) - zv_trd(A2D(0),:)
         CALL trd_dyn( zu_trd, zv_trd, jpdyn_keg, kt, Kmm )
         zu_trd(A2D(0),:) = puu(A2D(0),1:jpkm1,Krhs)
         zv_trd(A2D(0),:) = pvv(A2D(0),1:jpkm1,Krhs)
      ENDIF
      !
#define zFwu   zFu_t
#define zFwv   zFv_t
#define zww    zwu

!!gm TO BE tested : UNmasked horizontal trends and NO vertical component   !!!

      !                              ! ========================== !
      !                              !  Vertical advection k-slab !
      !                              ! ========================== !
      !
      !                              ! surface vertical fluxes
      !
      IF( PRESENT( pFw ) ) THEN
         zFw => pFw(:,:,1)
      ELSE
         zFw(T2D(1)) => zww(:,:)
         DO_2D( 0, 1, 0, 1 )
            zww(ji,jj) = e1e2t(ji,jj) * ww(ji,jj,1)
         END_2D
      ENDIF
      ! 
      IF( lk_linssh ) THEN                ! linear free surface: advection through the surface z=0
         DO_2D( 0, 0, 0, 0 )
            zFwu(ji,jj) = 0.5_wp * ( zFw(ji,jj) + zFw(ji+1,jj  ) ) * puu(ji,jj,1,Kmm)
            zFwv(ji,jj) = 0.5_wp * ( zFw(ji,jj) + zFw(ji  ,jj+1) ) * pvv(ji,jj,1,Kmm)
         END_2D
      ELSE                                ! non linear free: surface advective fluxes set to zero
         DO_2D( 0, 0, 0, 0 )
            zFwu(ji,jj) = 0._wp
            zFwv(ji,jj) = 0._wp
         END_2D
      ENDIF
      !
      DO jk = 1, jpk-2               !  divergence of advective fluxes
         !
         IF( PRESENT( pFw ) ) THEN        ! vertical transport at level k+1
            zFw => pFw(:,:,jk+1)
         ELSE
            DO_2D( 0, 1, 0, 1 )
               zFw(ji,jj) = e1e2t(ji,jj) * ww(ji,jj,jk+1)
            END_2D
         ENDIF
         !
         IF( PRESENT( pUe ) ) THEN       !-  2D RHS : MASKED vertically cumulated  -!
!!gm TO BE tested : UNmasked horizontal trends and NO vertical component   !!!
            DO_2D( 0, 0, 0, 0 )
               !                             ! vertical flux at level k+1
               zzFwu_kp1 = 0.25_wp * ( zFw(ji,jj) + zFw(ji+1,jj  ) ) * ( puu(ji,jj,jk+1,Kmm) + puu(ji,jj,jk,Kmm) )
               zzFwv_kp1 = 0.25_wp * ( zFw(ji,jj) + zFw(ji  ,jj+1) ) * ( pvv(ji,jj,jk+1,Kmm) + pvv(ji,jj,jk,Kmm) )
               !                             ! divergence of vertical momentum flux
               pUe(ji,jj) = pUe(ji,jj)  - ( zFwu(ji,jj) - zzFwu_kp1 ) * r1_e1e2u(ji,jj) * umask(ji,jj,jk)
               pVe(ji,jj) = pVe(ji,jj)  - ( zFwv(ji,jj) - zzFwv_kp1 ) * r1_e1e2v(ji,jj) * vmask(ji,jj,jk)
               !                            ! store vertical flux for next level calculation
               zFwu(ji,jj) = zzFwu_kp1
               zFwv(ji,jj) = zzFwv_kp1
            END_2D
         ELSE                            !-  added the 3D RHS  -!
            DO_2D( 0, 0, 0, 0 )
               !                             ! vertical flux at level k+1
               zzFwu_kp1 = 0.25_wp * ( zFw(ji,jj) + zFw(ji+1,jj  ) ) * ( puu(ji,jj,jk+1,Kmm) + puu(ji,jj,jk,Kmm) )
               zzFwv_kp1 = 0.25_wp * ( zFw(ji,jj) + zFw(ji  ,jj+1) ) * ( pvv(ji,jj,jk+1,Kmm) + pvv(ji,jj,jk,Kmm) )
               !                             ! divergence of vertical momentum flux
               puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zFwu(ji,jj) - zzFwu_kp1 ) * r1_e1e2u(ji,jj)   &
                  &                                      / e3u(ji,jj,jk,Kmm)
               pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zFwv(ji,jj) - zzFwv_kp1 ) * r1_e1e2v(ji,jj)   &
                  &                                      / e3v(ji,jj,jk,Kmm)
               !                             ! store vertical flux for next level calculation
               zFwu(ji,jj) = zzFwu_kp1
               zFwv(ji,jj) = zzFwv_kp1
            END_2D
         ENDIF
      END DO
      !
      jk = jpkm1                     ! Bottom last ocean level
      IF( PRESENT( pUe ) ) THEN          !-  2D RHS : MASKED vertically cumulated  -!
         DO_2D( 0, 0, 0, 0 )
            pUe(ji,jj) = pUe(ji,jj) - zFwu(ji,jj) * r1_e1e2u(ji,jj) * umask(ji,jj,jk)
            pVe(ji,jj) = pVe(ji,jj) - zFwv(ji,jj) * r1_e1e2v(ji,jj) * vmask(ji,jj,jk)
         END_2D
      ELSE                               !-  added the 3D RHS  -!
         DO_2D( 0, 0, 0, 0 )
            puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - zFwu(ji,jj) * r1_e1e2u(ji,jj)   &
               &                                      / e3u(ji,jj,jk,Kmm)
            pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - zFwv(ji,jj) * r1_e1e2v(ji,jj)   &
               &                                      / e3v(ji,jj,jk,Kmm)
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
      IF( l_trddyn ) THEN            ! trends: send trend to trddyn for diagnostic
         zu_trd(A2D(0),:) = puu(A2D(0),1:jpkm1,Krhs) - zu_trd(A2D(0),:)
         zv_trd(A2D(0),:) = pvv(A2D(0),1:jpkm1,Krhs) - zv_trd(A2D(0),:)
         CALL trd_dyn( zu_trd, zv_trd, jpdyn_zad, kt, Kmm )
         DEALLOCATE( zu_trd, zv_trd )
      ENDIF
      !
#undef zFwu
#undef zFwv
#undef zww
      !
#if ! defined key_PSYCLONE_2p5p0
      IF( .NOT.PRESENT( pFu ) )   DEALLOCATE( zwu, zwv )
#endif
      !
      !                                   ! Control print
      IF(sn_cfctl%l_prtctl)   CALL prt_ctl( tab3d_1=puu(:,:,:,Krhs), clinfo1=' cen2 adv - Ua: ', mask1=umask,   &
         &                                  tab3d_2=pvv(:,:,:,Krhs), clinfo2=           ' Va: ', mask2=vmask, clinfo3='dyn' )
      !
         !
   END SUBROUTINE dyn_adv_cen2

   !!==============================================================================
END MODULE dynadv_cen2
