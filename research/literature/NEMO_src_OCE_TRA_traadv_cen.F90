MODULE traadv_cen
   !!======================================================================
   !!                     ***  MODULE  traadv_cen  ***
   !! Ocean  tracers:   advective trend (2nd/4th order centered)
   !!======================================================================
   !! History :  3.7  ! 2014-05  (G. Madec)  original code
   !!            4.5  ! 2022-06  (S. Techene, G. Madec)  refactorization to reduce local memory usage
   !!            5.x  ! 2026-03  (S. Griffies, G. Madec)  thickness weighted tracer tendency 
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   tra_adv_cen   : update the tracer trend with the advection trends using a centered or scheme (2nd or 4th order)
   !!                   NB: on the vertical it is actually a 4th order COMPACT scheme which is used
   !!----------------------------------------------------------------------
   USE dom_oce        ! ocean space and time domain
   USE eosbn2         ! equation of state
   USE traadv_fct     ! acces to routine interp_4th_cpt
   USE trd_oce        ! trends: ocean variables
   USE trdtra         ! trends manager: tracers
   USE diaptr         ! poleward transport diagnostics
   USE diaar5         ! AR5 diagnostics
   USE domutl, ONLY : lbnd_ij
   !
   USE in_out_manager ! I/O manager
   USE iom            ! IOM library
   USE trc_oce        ! share passive tracers/Ocean variables
   USE lib_mpp        ! MPP library

   IMPLICIT NONE
   PRIVATE

   PUBLIC   tra_adv_cen        ! called by traadv.F90

   REAL(wp) ::   r1_6 = 1._wp / 6._wp   ! =1/6

   LOGICAL ::   l_trd   ! flag to compute trends
   LOGICAL ::   l_ptr   ! flag to compute poleward transport
   LOGICAL ::   l_hst   ! flag to compute heat/salt transport

   REAL(wp), PUBLIC, PARAMETER :: pp_stb_thres_tra_ce2 = 1.35_wp  ! starting Courant number threshold for adaptive implicit vertical advection
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_cstra_tra_ce2 = 1.60_wp  ! stability constraint for cen2 with RK3 
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_thres_tra_ce4 = 0.85_wp  ! starting Courant number threshold for adaptive implicit vertical advection
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_cstra_tra_ce4 = 1.00_wp  ! stability constraint for cen4 with RK3 
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_thres_tra_co4 = 0.85_wp  ! starting Courant number threshold for adaptive implicit vertical advection
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_cstra_tra_co4 = 1.00_wp  ! stability constraint for compact4 with RK3

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE tra_adv_cen( kt, kit000, cdtype, pU, pV, pW,     &
      &                    Kmm, pt, kjpt, Krhs, kn_cen_h, kn_cen_v )
      !!
      INTEGER                                  , INTENT(in   ) ::   kt              ! ocean time-step index
      INTEGER                                  , INTENT(in   ) ::   Kmm, Krhs       ! ocean time level indices
      INTEGER                                  , INTENT(in   ) ::   kit000          ! first time step index
      CHARACTER(len=3)                         , INTENT(in   ) ::   cdtype          ! =TRA or TRC (tracer indicator)
      INTEGER                                  , INTENT(in   ) ::   kjpt            ! number of tracers
      INTEGER                                  , INTENT(in   ) ::   kn_cen_h        ! =2/4 (2nd or 4th order scheme)
      INTEGER                                  , INTENT(in   ) ::   kn_cen_v        ! =2/4 (2nd or 4th order scheme)
      REAL(wp), DIMENSION(:,:,:               ), INTENT(in   ) ::   pU, pV, pW      ! 3 ocean volume flux components   [m^3/s]
      REAL(wp), DIMENSION(jpi,jpj,jpk,kjpt,jpt), INTENT(inout) ::   pt              ! tracers and RHS of tracer equation
      !!
      CALL tra_adv_cen_t( kt, kit000, cdtype, pU, pV, pW, lbnd_ij(pU),  &
         &                Kmm, pt, kjpt, Krhs, kn_cen_h, kn_cen_v       )
   END SUBROUTINE tra_adv_cen

   SUBROUTINE tra_adv_cen_t( kt, kit000, cdtype, pU, pV, pW, ktpuvw, &
      &                      Kmm, pt, kjpt, Krhs, kn_cen_h, kn_cen_v )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE tra_adv_cen  ***
      !!
      !! ** Purpose :   Compute the now trend due to the advection of tracers
      !!      and add it to the general trend of passive tracer equations.
      !!
      !! ** Method  :   The advection is evaluated by a 2nd or 4th order scheme
      !!               using middle fields.
      !!       kn_cen_h = 2  ==>> 2nd order centered scheme on the horizontal
      !!                = 4  ==>> 4th order    -        -       -      -
      !!       kn_cen_v = 2  ==>> 2nd order centered scheme on the vertical
      !!                = 4  ==>> 4th order COMPACT  scheme     -      -
      !!
      !! ** Action : - update pt(:,:,:,:,Krhs)  with the now advective tracer trends
      !!             - send trends to trdtra module for further diagnostcs (l_trdtra=T)
      !!             - poleward advective heat and salt transport (l_diaptr=T)
      !!----------------------------------------------------------------------
      INTEGER,  DIMENSION(2)                   , INTENT(in   ) ::   ktpuvw
      INTEGER                                  , INTENT(in   ) ::   kt              ! ocean time-step index
      INTEGER                                  , INTENT(in   ) ::   Kmm, Krhs       ! ocean time level indices
      INTEGER                                  , INTENT(in   ) ::   kit000          ! first time step index
      CHARACTER(len=3)                         , INTENT(in   ) ::   cdtype          ! =TRA or TRC (tracer indicator)
      INTEGER                                  , INTENT(in   ) ::   kjpt            ! number of tracers
      INTEGER                                  , INTENT(in   ) ::   kn_cen_h        ! =2/4 (2nd or 4th order scheme)
      INTEGER                                  , INTENT(in   ) ::   kn_cen_v        ! =2/4 (2nd or 4th order scheme)
      REAL(wp), DIMENSION(AB2D(ktpuvw),JPK    ), INTENT(in   ) ::   pU, pV, pW      ! 3 ocean volume flux components
      REAL(wp), DIMENSION(jpi,jpj,jpk,kjpt,jpt), INTENT(inout) ::   pt              ! tracers and RHS of tracer equation
      !
      INTEGER  ::   ji, jj, jk, jn   ! dummy loop indices
      INTEGER  ::   ierr             ! local integer
      REAL(wp) ::   zC2t_u, zC4t_u   ! local scalars
      REAL(wp) ::   zC2t_v, zC4t_v   !   -      -
      REAL(wp) ::   ztFw_kp1
      REAL(wp), DIMENSION(T2D(1))              ::   ztFu, ztFv
#if ! defined key_PSYCLONE_2p5p0
      REAL(wp), DIMENSION(:,:)   , ALLOCATABLE ::   ztu, ztv
      REAL(wp), DIMENSION(:,:,:) , ALLOCATABLE ::   ztw
#else
      REAL(wp), DIMENSION(T2D(2))              ::   ztu, ztv
      REAL(wp), DIMENSION(T2D(1),jpk)          ::   ztw
#endif
      !!----------------------------------------------------------------------
      !
      IF( .NOT. l_istiled .OR. ntile == 1 )  THEN                       ! Do only on the first tile
         IF( kt == kit000 )  THEN
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) 'tra_adv_cen : centered advection scheme on ', cdtype, ' order h/v =', kn_cen_h,'/', kn_cen_v
            IF(lwp) WRITE(numout,*) '~~~~~~~~~~~~ '
         ENDIF
         !                          ! set local switches
         l_trd = .FALSE.
         l_hst = .FALSE.
         l_ptr = .FALSE.
         IF( ( cdtype == 'TRA' .AND. l_trdtra ) .OR. ( cdtype == 'TRC' .AND. l_trdtrc ) )                      l_trd = .TRUE.
         IF(  l_diaptr .AND. cdtype == 'TRA' .AND. ( iom_use( 'sophtadv' )  .OR. iom_use( 'sopstadv'  ) )  )   l_ptr = .TRUE.
         IF(  l_diaar5 .AND. cdtype == 'TRA' .AND. ( iom_use("uadv_heattr") .OR. iom_use("vadv_heattr") .OR. &
            &                                        iom_use("uadv_salttr") .OR. iom_use("vadv_salttr") )  )   l_hst = .TRUE.
      ENDIF
      !
#if ! defined key_PSYCLONE_2p5p0
      IF( kn_cen_h == 4 )   ALLOCATE( ztu(T2D(2)) , ztv(T2D(2)) )   ! horizontal 4th order only
      IF( kn_cen_v == 4 )   ALLOCATE( ztw(T2D(1),jpk) )             ! vertical   4th order only
#endif
      !
      DO jn = 1, kjpt            !==  loop over the tracers  ==!
         !
         SELECT CASE( kn_cen_h )       !--  Horizontal divergence of advective fluxes  --!
         !
!!st limitation : does not take into acccount iceshelf specificity
!!                in case of linssh
         CASE(  2  )                         !* 2nd order centered
            DO jk = 1, jpkm1
               !
               DO_2D( 1, 0, 1, 0 )                     ! Horizontal fluxes at layer jk
                  ztFu(ji,jj) = 0.5_wp * pU(ji,jj,jk) * ( pt(ji,jj,jk,jn,Kmm) + pt(ji+1,jj  ,jk,jn,Kmm) )
                  ztFv(ji,jj) = 0.5_wp * pV(ji,jj,jk) * ( pt(ji,jj,jk,jn,Kmm) + pt(ji  ,jj+1,jk,jn,Kmm) )
               END_2D
               !
               DO_2D( 0, 0, 0, 0 )                     ! Horizontal divergence of advective fluxes
                  pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) - (  ( ztFu(ji,jj) - ztFu(ji-1,jj  ) )    &   ! add () for NP repro
                     &                                           + ( ztFv(ji,jj) - ztFv(ji  ,jj-1) )  ) * r1_e1e2t(ji,jj)  
               END_2D
               !                                 ! "Poleward" heat and salt transports
               IF( l_ptr )   CALL dia_ptr_hst( jn, 'adv', ztFv(:,:) )
               !                                 !  heat and salt transport
               IF( l_hst )   CALL dia_ar5_hst( jn, 'adv', ztFu(:,:), ztFv(:,:), ldfin=(jk == jpkm1) )
            END DO
            !
         CASE(  4  )                         !* 4th order centered
            DO jk = 1, jpkm1
               DO_2D( 2, 1, 2, 1 )          ! masked gradient
                  ztu(ji,jj) = ( pt(ji+1,jj  ,jk,jn,Kmm) - pt(ji,jj,jk,jn,Kmm) ) * umask(ji,jj,jk)
                  ztv(ji,jj) = ( pt(ji  ,jj+1,jk,jn,Kmm) - pt(ji,jj,jk,jn,Kmm) ) * vmask(ji,jj,jk)
               END_2D
               !
               DO_2D( 1, 0, 1, 0 )                    ! Horizontal advective fluxes
                  zC2t_u = pt(ji,jj,jk,jn,Kmm) + pt(ji+1,jj  ,jk,jn,Kmm)   ! C2 interpolation of T at u- & v-points (x2)
                  zC2t_v = pt(ji,jj,jk,jn,Kmm) + pt(ji  ,jj+1,jk,jn,Kmm)
                  !                                                        ! C4 interpolation of T at u- & v-points (x2)
                  zC4t_u =  zC2t_u + r1_6 * ( ztu(ji-1,jj  ) - ztu(ji+1,jj  ) )
                  zC4t_v =  zC2t_v + r1_6 * ( ztv(ji  ,jj-1) - ztv(ji  ,jj+1) )
                  !                                                        ! C4 fluxes
                  ztFu(ji,jj) =  0.5_wp * pU(ji,jj,jk) * zC4t_u
                  ztFv(ji,jj) =  0.5_wp * pV(ji,jj,jk) * zC4t_v
               END_2D
               !
               DO_2D( 0, 0, 0, 0 )                                         ! Horizontal divergence of advective fluxes
                  pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) - (  ( ztFu(ji,jj) - ztFu(ji-1,jj  ) )    &   ! add () for NP repro
                     &                                           + ( ztFv(ji,jj) - ztFv(ji  ,jj-1) )  ) * r1_e1e2t(ji,jj)  
               END_2D
               !                                 ! "Poleward" heat and salt transports
               IF( l_ptr )   CALL dia_ptr_hst( jn, 'adv', ztFv(:,:) )
               !                                 !  heat and salt transport
               IF( l_hst )   CALL dia_ar5_hst( jn, 'adv', ztFu(:,:), ztFv(:,:), ldfin=(jk == jpkm1) )
            END DO
            !
         CASE DEFAULT
            CALL ctl_stop( 'traadv_cen: wrong value for nn_cen' )
         END SELECT
         !
#define ztFw  ztFu
         !
         IF( lk_linssh ) THEN                !* top value   (linear free surf. only as zwz is multiplied by wmask)
            DO_2D( 0, 0, 0, 0 )
               ztFw(ji,jj) = pW(ji,jj,1) * pt(ji,jj,1,jn,Kmm)
            END_2D
         ELSE
            DO_2D( 0, 0, 0, 0 )
               ztFw(ji,jj) = 0._wp
            END_2D
         ENDIF
         !
         SELECT CASE( kn_cen_v )       !--  Vertical divergence of advective fluxes  --!   (interior)
         !
         CASE(  2  )                         !* 2nd order centered
            DO jk = 1, jpk-2
               DO_2D( 0, 0, 0, 0 )                             ! Vertical fluxes
                  ztFw_kp1 = 0.5_wp * pW(ji,jj,jk+1) * ( pt(ji,jj,jk+1,jn,Kmm) + pt(ji,jj,jk,jn,Kmm) ) * wmask(ji,jj,jk+1)
                  !
                  pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) - (  ztFw(ji,jj) - ztFw_kp1  ) * r1_e1e2t(ji,jj) 
                  ztFw(ji,jj) = ztFw_kp1
               END_2D
            END DO
            jk = jpkm1                                         ! bottom vertical flux set to zero for all tracers
            DO_2D( 0, 0, 0, 0 )
               pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) - ztFw(ji,jj) * r1_e1e2t(ji,jj)  
            END_2D
            !
         CASE(  4  )                         !* 4th order compact
            CALL interp_4th_cpt( pt(:,:,:,jn,Kmm) , ztw )      ! ztw = interpolated value of T at w-point
            !
            DO jk = 1, jpk-2
               !
               DO_2D( 0, 0, 0, 0 )
                  ztFw_kp1 = pW(ji,jj,jk+1) * ztw(ji,jj,jk+1) * wmask(ji,jj,jk+1)
                  !                          ! Divergence of advective fluxes
                  pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) - (  ztFw(ji,jj) - ztFw_kp1  ) * r1_e1e2t(ji,jj)  
                  !                          ! update
                  ztFw(ji,jj) = ztFw_kp1
               END_2D
               !
            END DO
            !
            jk = jpkm1       ! bottom vertical flux set to zero for all tracers
            DO_2D( 0, 0, 0, 0 )
               pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) - ztFw(ji,jj) * r1_e1e2t(ji,jj) 
            END_2D
            !
         END SELECT
         !
#undef ztFw
         !                               ! trend diagnostics
!!gm + !!st to be done with the whole rewritting of trd
!!          trd routine arguments MUST be changed adding jk and zwx, zwy in 2D
!!         IF( l_trd ) THEN
!!            CALL trd_tra( kt, Kmm, Krhs, cdtype, jn, jptra_xad, zwx, pU, pt(:,:,:,jn,Kmm) )
!!            CALL trd_tra( kt, Kmm, Krhs, cdtype, jn, jptra_yad, zwy, pV, pt(:,:,:,jn,Kmm) )
!!            CALL trd_tra( kt, Kmm, Krhs, cdtype, jn, jptra_zad, zwz, pW, pt(:,:,:,jn,Kmm) )
!!         ENDIF
         !
      END DO
      !
#if ! defined key_PSYCLONE_2p5p0
      IF( kn_cen_h == 4 )   DEALLOCATE( ztu , ztv )   ! horizontal 4th order only
      IF( kn_cen_v == 4 )   DEALLOCATE( ztw )             ! vertical   4th order only
#endif
      !
   END SUBROUTINE tra_adv_cen_t

   !!======================================================================
END MODULE traadv_cen
