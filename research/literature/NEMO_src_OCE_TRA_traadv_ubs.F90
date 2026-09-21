MODULE traadv_ubs
   !!==============================================================================
   !!                       ***  MODULE  traadv_ubs  ***
   !! Ocean active tracers:  horizontal & vertical advective trend
   !!==============================================================================
   !! History :  1.0  !  2006-08  (L. Debreu, R. Benshila)  Original code
   !!            3.3  !  2010-05  (C. Ethe, G. Madec)  merge TRC-TRA + switch from velocity to transport
   !!            5.x  !  2026-03  (S. Griffies, G. Madec)  thickness weighted tracer tendency 
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   tra_adv_ubs : update the tracer trend with the horizontal
   !!                 advection trends using a third order biaised scheme
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and active tracers
   USE dom_oce        ! ocean space and time domain
   USE trc_oce        ! share passive tracers/Ocean variables
   USE trd_oce        ! trends: ocean variables
   USE traadv_fct     ! acces to routine interp_4th_cpt
   USE trdtra         ! trends manager: tracers
   USE diaptr         ! poleward transport diagnostics
   USE diaar5         ! AR5 diagnostics
   USE domutl, ONLY : lbnd_ij
   !
   USE iom            ! I/O library
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! massively parallel library
   USE lbclnk         ! ocean lateral boundary condition (or mpp link)
   USE lib_fortran    ! Fortran utilities (allows no signed zero when 'key_nosignedzero' defined)

   IMPLICIT NONE
   PRIVATE

   PUBLIC   tra_adv_ubs   ! routine called by traadv module

   LOGICAL :: l_trd   ! flag to compute trends
   LOGICAL :: l_ptr   ! flag to compute poleward transport
   LOGICAL :: l_hst   ! flag to compute heat transport
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_thres_tra_ubs = 1.35_wp  ! starting Courant number threshold for adaptive implicit vertical advection
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_cstra_tra_ubs = 1.60_wp  ! stability constraint


   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE tra_adv_ubs( kt, kit000, cdtype, pdt, pU, pV, pW,          &
   &                       Kbb, Kmm, pt, kjpt, Krhs, kn_ubs_v )
      !!
      INTEGER                                  , INTENT(in   ) ::   kt              ! ocean time-step index
      INTEGER                                  , INTENT(in   ) ::   Kbb, Kmm, Krhs  ! ocean time level indices
      INTEGER                                  , INTENT(in   ) ::   kit000          ! first time step index
      CHARACTER(len=3)                         , INTENT(in   ) ::   cdtype          ! =TRA or TRC (tracer indicator)
      INTEGER                                  , INTENT(in   ) ::   kjpt            ! number of tracers
      INTEGER                                  , INTENT(in   ) ::   kn_ubs_v        ! number of tracers
      REAL(wp)                                 , INTENT(in   ) ::   pdt            ! tracer time-step
      REAL(wp), DIMENSION(:,:,:               ), INTENT(in   ) ::   pU, pV, pW      ! 3 ocean volume transport components
      REAL(wp), DIMENSION(jpi,jpj,jpk,kjpt,jpt), INTENT(inout) ::   pt              ! tracers and RHS of tracer equation
      !!
      CALL tra_adv_ubs_t( kt, kit000, cdtype, pdt, pU, pV, pW, lbnd_ij(pU),  &
         &                Kbb, Kmm, pt, kjpt, Krhs, kn_ubs_v                 )
   END SUBROUTINE tra_adv_ubs

   SUBROUTINE tra_adv_ubs_t( kt, kit000, cdtype, pdt, pU, pV, pW, ktpuvw, &
      &                      Kbb, Kmm, pt, kjpt, Krhs, kn_ubs_v           )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE tra_adv_ubs  ***
      !!
      !! ** Purpose :   Compute the now trend due to the advection of tracers
      !!      and add it to the general trend of passive tracer equations.
      !!
      !! ** Method  :   The 3rd order Upstream Biased Scheme (UBS) is based on an
      !!      upstream-biased parabolic interpolation (Shchepetkin and McWilliams 2005)
      !!      It is only used in the horizontal direction.
      !!      For example the i-component of the advective fluxes are given by :
      !!                !  e2u e3u un ( mi(Tn) - zltu(i  ) )   if un(i) >= 0
      !!          ztu = !  or
      !!                !  e2u e3u un ( mi(Tn) - zltu(i+1) )   if un(i) < 0
      !!      where zltu is the second derivative of the before temperature field:
      !!          zltu = 1/e3t di[ e2u e3u / e1u di[Tb] ]
      !!        This results in a dissipatively dominant (i.e. hyper-diffusive)
      !!      truncation error. The overall performance of the advection scheme
      !!      is similar to that reported in (Farrow and Stevens, 1995).
      !!        For stability reasons, the first term of the fluxes which corresponds
      !!      to a second order centered scheme is evaluated using the now velocity
      !!      (centered in time) while the second term which is the diffusive part
      !!      of the scheme, is evaluated using the before velocity (forward in time).
      !!      Note that UBS is not positive. Do not use it on passive tracers.
      !!                On the vertical, the advection is evaluated using a FCT scheme,
      !!      as the UBS have been found to be too diffusive.
      !!                kn_ubs_v argument controles whether the FCT is based on
      !!      a 2nd order centrered scheme (kn_ubs_v=2) or on a 4th order compact
      !!      scheme (kn_ubs_v=4).
      !!
      !! ** Action : - update pt(:,:,:,:,Krhs)  with the now advective tracer trends
      !!             - send trends to trdtra module for further diagnostcs (l_trdtra=T)
      !!             - poleward advective heat and salt transport (ln_diaptr=T)
      !!
      !! Reference : Shchepetkin, A. F., J. C. McWilliams, 2005, Ocean Modelling, 9, 347-404.
      !!             Farrow, D.E., Stevens, D.P., 1995, J. Phys. Ocean. 25, 1731Ã¯Â¿Â½1741.
      !!----------------------------------------------------------------------
      INTEGER,  DIMENSION(2)                   , INTENT(in   ) ::   ktpuvw
      INTEGER                                  , INTENT(in   ) ::   kt              ! ocean time-step index
      INTEGER                                  , INTENT(in   ) ::   Kbb, Kmm, Krhs  ! ocean time level indices
      INTEGER                                  , INTENT(in   ) ::   kit000          ! first time step index
      CHARACTER(len=3)                         , INTENT(in   ) ::   cdtype          ! =TRA or TRC (tracer indicator)
      INTEGER                                  , INTENT(in   ) ::   kjpt            ! number of tracers
      INTEGER                                  , INTENT(in   ) ::   kn_ubs_v        ! number of tracers
      REAL(wp)                                 , INTENT(in   ) ::   pdt             ! tracer time-step
      REAL(wp), DIMENSION(AB2D(ktpuvw),JPK    ), INTENT(in   ) ::   pU, pV, pW      ! 3 ocean volume transport components
      REAL(wp), DIMENSION(jpi,jpj,jpk,kjpt,jpt), INTENT(inout) ::   pt              ! tracers and RHS of tracer equation
      !
      INTEGER  ::   ji, jj, jk, jn   ! dummy loop indices
      REAL(wp) ::   ztra, zcoef                              ! local scalars
      REAL(wp) ::   zfp_ui, zfm_ui, zcenut, zfp_wk, zfm_wk   !   -      -
      REAL(wp) ::   zfp_vj, zfm_vj, zcenvt, zeeu, zeev       !   -      -
      REAL(wp), DIMENSION(T2D(nn_hls),jpk) ::   zti, ztFw              ! 3D workspace
      REAL(wp), DIMENSION(T2D(2)) ::   ztu, ztv, zltu, zltv, ztFu, ztFv ! 2D workspace
#if ! defined key_PSYCLONE_2p5p0
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   ztw
#else
      REAL(wp), DIMENSION(T2D(1),jpk)         ::   ztw
#endif
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   ztrdx, ztrdy, ztrdz
      !!----------------------------------------------------------------------
      !
      IF( .NOT. l_istiled .OR. ntile == 1 )  THEN                       ! Do only on the first tile
         IF( kt == kit000 )  THEN
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) 'tra_adv_ubs :  horizontal UBS advection scheme on ', cdtype
            IF(lwp) WRITE(numout,*) '~~~~~~~~~~~~'
         ENDIF
         !
         l_trd = .FALSE.
         l_hst = .FALSE.
         l_ptr = .FALSE.
         IF( ( cdtype == 'TRA' .AND. l_trdtra ) .OR. ( cdtype == 'TRC' .AND. l_trdtrc ) )                      l_trd = .TRUE.
         IF(  l_diaptr .AND. cdtype == 'TRA' .AND. ( iom_use( 'sophtadv' )  .OR. iom_use( 'sopstadv'  ) )  )   l_ptr = .TRUE.
         IF(  l_diaar5 .AND. cdtype == 'TRA' .AND. ( iom_use("uadv_heattr") .OR. iom_use("vadv_heattr") .OR. &
            &                                        iom_use("uadv_salttr") .OR. iom_use("vadv_salttr") )  )   l_hst = .TRUE.
      ENDIF
      !
      IF( l_trd .OR. l_hst .OR. l_ptr )  THEN
         ALLOCATE( ztrdx(T2D(1),jpk), ztrdy(T2D(1),jpk), ztrdz(T2D(0),jpk) )
         ztrdx(:,:,jpk) = 0._wp   ;   ztrdy(:,:,jpk) = 0._wp
         ztrdz(:,:,1  ) = 0._wp   ;   ztrdz(:,:,jpk) = 0._wp
      ENDIF
      !
      ztFw(:,:,jpk) = 0._wp
      zti (:,:,jpk) = 0._wp
      !                                                          ! ===========
      DO jn = 1, kjpt                                            ! tracer loop
         !                                                       ! ===========
         DO jk = 1, jpkm1      
            !                     !==  horizontal laplacian of before tracer ==!
            !
            DO_2D( 2, 1, 2, 1 )                   ! First derivative (masked gradient)
               zeeu = e2_e1u(ji,jj) * e3u(ji,jj,jk,Kmm) * umask(ji,jj,jk)
               zeev = e1_e2v(ji,jj) * e3v(ji,jj,jk,Kmm) * vmask(ji,jj,jk)
               ztu(ji,jj) = zeeu * ( pt(ji+1,jj  ,jk,jn,Kbb) - pt(ji,jj,jk,jn,Kbb) )
               ztv(ji,jj) = zeev * ( pt(ji  ,jj+1,jk,jn,Kbb) - pt(ji,jj,jk,jn,Kbb) )
            END_2D
            DO_2D( 1, 1, 1, 1 )                   ! Second derivative (divergence)
               zcoef = 1._wp / ( 6._wp * e3t(ji,jj,jk,Kmm) )
               zltu(ji,jj) = (  ztu(ji,jj) - ztu(ji-1,jj)  ) * zcoef
               zltv(ji,jj) = (  ztv(ji,jj) - ztv(ji,jj-1)  ) * zcoef
            END_2D
            !
            DO_2D( 1, 0, 1, 0 )   !==  Horizontal advective fluxes  ==!     (UBS)
               zfp_ui = pU(ji,jj,jk) + ABS( pU(ji,jj,jk) )        ! upstream transport (x2)
               zfm_ui = pU(ji,jj,jk) - ABS( pU(ji,jj,jk) )
               zfp_vj = pV(ji,jj,jk) + ABS( pV(ji,jj,jk) )
               zfm_vj = pV(ji,jj,jk) - ABS( pV(ji,jj,jk) )
               !                                                  ! 2nd order centered advective fluxes (x2)
               zcenut = pU(ji,jj,jk) * ( pt(ji,jj,jk,jn,Kmm) + pt(ji+1,jj  ,jk,jn,Kmm) )
               zcenvt = pV(ji,jj,jk) * ( pt(ji,jj,jk,jn,Kmm) + pt(ji  ,jj+1,jk,jn,Kmm) )
               !                                                  ! UBS advective fluxes
               ztFu(ji,jj) = 0.5 * ( zcenut - ( zfp_ui * zltu(ji,jj) + zfm_ui * zltu(ji+1,jj) ) )   ! add () for NP repro
               ztFv(ji,jj) = 0.5 * ( zcenvt - ( zfp_vj * zltv(ji,jj) + zfm_vj * zltv(ji,jj+1) ) )
            END_2D
            !
            !                     !==  add the horizontal advective trend  ==!
            DO_2D( 0, 0, 0, 0 )
               ztra = - (  ( ztFu(ji,jj) - ztFu(ji-1,jj  ) )   &   ! add () for NP reproducibility
                  &      + ( ztFv(ji,jj) - ztFv(ji  ,jj-1) ) ) * r1_e1e2t(ji,jj) 
               !
               pt(ji,jj,jk,jn,Krhs) =   pt(ji,jj,jk,jn,Krhs) +       ztra   * tmask(ji,jj,jk)
               !
               zti(ji,jj,jk)        = ( pt(ji,jj,jk,jn,Kbb ) + pdt * ztra / e3t(ji,jj,jk,Kmm) ) * tmask(ji,jj,jk)
            END_2D
            !
            IF( l_trd .OR. l_hst .OR. l_ptr ) THEN   ! trend diagnostics // heat/salt transport
               DO_2D( 1, 0, 1, 0 )
                  ztrdx(ji,jj,jk) = ztFu(ji,jj)
                  ztrdy(ji,jj,jk) = ztFv(ji,jj)
               END_2D
            ENDIF
            !
         END DO
         !
         !                       !== vertical advective trend  ==!
         !
         SELECT CASE( kn_ubs_v )       ! select the vertical advection scheme
         !
         CASE(  2  )                   ! 2nd order FCT
            !
            !                               !==  upstream advection with initial mass fluxes & intermediate update  ==!
            DO_3D( 0, 0, 0, 0, 2, jpkm1 )
               zfp_wk = pW(ji,jj,jk) + ABS( pW(ji,jj,jk) )
               zfm_wk = pW(ji,jj,jk) - ABS( pW(ji,jj,jk) )
               ztFw(ji,jj,jk) = 0.5_wp * ( zfp_wk * pt(ji,jj,jk,jn,Kbb) + zfm_wk * pt(ji,jj,jk-1,jn,Kbb) ) * wmask(ji,jj,jk)
            END_3D
            IF( lk_linssh ) THEN                ! top ocean value (only in linear free surface as ztFw has been w-masked)
               IF( ln_isfcav ) THEN                   ! top of the ice-shelf cavities and at the ocean surface
                  DO_2D( 0, 0, 0, 0 )
                     ztFw(ji,jj, mikt(ji,jj) ) = pW(ji,jj,mikt(ji,jj)) * pt(ji,jj,mikt(ji,jj),jn,Kbb)   ! linear free surface
                  END_2D
               ELSE                                   ! no cavities: only at the ocean surface
                  DO_2D( 0, 0, 0, 0 )
                     ztFw(ji,jj,1) = pW(ji,jj,1) * pt(ji,jj,1,jn,Kbb)
                  END_2D
               ENDIF
            ELSE
               DO_2D( 0, 0, 0, 0 )
                  ztFw(ji,jj,1) = 0._wp
               END_2D
            ENDIF
            !
            IF( l_trd ) THEN
               DO_3D( 0, 0, 0, 0, 2, jpkm1 )
                  ztrdz(ji,jj,jk) = ztFw(ji,jj,jk) ! trend diagnostics (contribution of upstream fluxes)
               END_3D
            ENDIF
            !
            !                               !== trend and after field with monotonic scheme ==!
            DO_3D( 0, 0, 0, 0, 1, jpkm1 )
               ztra = - ( ztFw(ji,jj,jk) - ztFw(ji,jj,jk+1) ) * r1_e1e2t(ji,jj) 
               !
               pt (ji,jj,jk,jn,Krhs) =   pt (ji,jj,jk,jn,Krhs) +       ztra   * tmask(ji,jj,jk)
               zti(ji,jj,jk)         = ( zti(ji,jj,jk)         + pdt * ztra / e3t(ji,jj,jk,Kmm) ) * tmask(ji,jj,jk) 
            END_3D
            !                               !==  anti-diffusive flux : high order minus low order ==!
            DO_3D( 0, 0, 0, 0, 2, jpkm1 )
               ztFw(ji,jj,jk) = ( 0.5_wp * pW(ji,jj,jk) * ( pt(ji,jj,jk,jn,Kmm) + pt(ji,jj,jk-1,jn,Kmm) ) - ztFw(ji,jj,jk) ) &
                  &             * wmask(ji,jj,jk)
            END_3D
            !                                            ! top ocean value: high order == upstream  ==>>  zwz=0
            IF( lk_linssh )   ztFw(:,:,1) = 0._wp        ! only ocean surface as interior zwz values have been w-masked
            !
            CALL nonosc_z( Kmm, pt(:,:,:,jn,Kbb), ztFw, zti, pdt )      !  monotonicity algorithm
            !
            IF( l_trd ) THEN
               DO_3D( 0, 0, 0, 0, 2, jpkm1 )
                  ztrdz(ji,jj,jk) = ztrdz(ji,jj,jk) + ztFw(ji,jj,jk) ! trend diagnostics add anti-diffusive fluxes to upstream fluxes
               END_3D
            ENDIF
            !
         CASE(  4  )                               ! 4th order COMPACT
            !
#if ! defined key_PSYCLONE_2p5p0
            ALLOCATE( ztw(T2D(1),jpk) )
#endif
            !
            CALL interp_4th_cpt( pt(:,:,:,jn,Kmm) , ztw )         ! 4th order compact interpolation of T at w-point
            DO_3D( 0, 0, 0, 0, 2, jpkm1 )
               ztFw(ji,jj,jk) = pW(ji,jj,jk) * ztw(ji,jj,jk) * wmask(ji,jj,jk)
            END_3D
            !
#if ! defined key_PSYCLONE_2p5p0
            DEALLOCATE( ztw )
#endif
            !
            IF( lk_linssh ) THEN
               DO_2D( 0, 0, 0, 0 )
                  ztFw(ji,jj,1) = pW(ji,jj,1) * pt(ji,jj,1,jn,Kmm)     !!gm ISF & 4th COMPACT doesn't work
               END_2D
            ELSE
               DO_2D( 0, 0, 0, 0 )
                  ztFw(ji,jj,1) = 0._wp
               END_2D
            ENDIF
            !
            IF( l_trd ) THEN
               DO_3D( 0, 0, 0, 0, 2, jpkm1 )
                  ztrdz(ji,jj,jk) = ztFw(ji,jj,jk) ! trend diagnostics
               END_3D
            ENDIF
            !
         END SELECT
         !
         DO_3D( 0, 0, 0, 0, 1, jpkm1 )   !  final trend with corrected fluxes
            ztra = - ( ztFw(ji,jj,jk) - ztFw(ji,jj,jk+1) ) * r1_e1e2t(ji,jj)
            pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) + ztra * tmask(ji,jj,jk)
         END_3D
         !
         IF( l_trd ) THEN              ! trend diagnostics
            CALL trd_tra( kt, Kmm, Krhs, cdtype, jn, jptra_xad, ztrdx, pU, pt(:,:,:,jn,Kmm) )
            CALL trd_tra( kt, Kmm, Krhs, cdtype, jn, jptra_yad, ztrdy, pV, pt(:,:,:,jn,Kmm) )
            CALL trd_tra( kt, Kmm, Krhs, cdtype, jn, jptra_zad, ztrdz, pW, pt(:,:,:,jn,Kmm) )
         ENDIF
         IF( l_hst )   CALL dia_ar5_hst( jn, 'adv', ztrdx(:,:,:), ztrdy(:,:,:) ) ! heat/salt transport
         IF( l_ptr )   CALL dia_ptr_hst( jn, 'adv', ztrdy(:,:,:) )            ! "Poleward" transports
         !
      END DO
      !
      IF( l_trd .OR. l_hst .OR. l_ptr )   DEALLOCATE( ztrdx, ztrdy, ztrdz )
      !
   END SUBROUTINE tra_adv_ubs_t


   SUBROUTINE nonosc_z( Kmm, pbef, pcc, paft, pdt )
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE nonosc_z  ***
      !!
      !! **  Purpose :   compute monotonic tracer fluxes from the upstream
      !!       scheme and the before field by a nonoscillatory algorithm
      !!
      !! **  Method  :   ... ???
      !!       warning : pbef and paft must be masked, but the boundaries
      !!       conditions on the fluxes are not necessary zalezak (1979)
      !!       drange (1995) multi-dimensional forward-in-time and upstream-
      !!       in-space based differencing for fluid
      !!----------------------------------------------------------------------
      INTEGER , INTENT(in   )                             ::   Kmm    ! time level index
      REAL(wp), INTENT(in   )                             ::   pdt    ! tracer time-step
      REAL(wp), INTENT(inout), DIMENSION(jpi,jpj,jpk)     ::   pbef   ! before field
      REAL(wp), INTENT(inout), DIMENSION(T2D(nn_hls),jpk) ::   paft   ! after field
      REAL(wp), INTENT(inout), DIMENSION(T2D(nn_hls),jpk) ::   pcc    ! monotonic flux in the k direction
      !
      INTEGER  ::   ji, jj, jk   ! dummy loop indices
      INTEGER  ::   ikm1         ! local integer
      REAL(wp) ::   zpos, zneg, zbt, za, zb, zc, zbig, zrtrn   ! local scalars
      REAL(wp), DIMENSION(T2D(0),jpk) ::   zbetup, zbetdo ! 3D workspace
      !!----------------------------------------------------------------------
      !
      zbig  = 1.e+15_wp   ! works ok with simple/double precison
      zrtrn = 1.e-15_wp
      zbetup(:,:,:) = 0._wp   ;   zbetdo(:,:,:) = 0._wp
      !
      ! Search local extrema
      ! --------------------
      !                    ! large negative value (-zbig) inside land
      DO_3D( 0, 0, 0, 0, 1, jpk )
         pbef(ji,jj,jk) = pbef(ji,jj,jk) * tmask(ji,jj,jk) - zbig * ( 1.e0 - tmask(ji,jj,jk) )
         paft(ji,jj,jk) = paft(ji,jj,jk) * tmask(ji,jj,jk) - zbig * ( 1.e0 - tmask(ji,jj,jk) )
      END_3D
      !
      DO jk = 1, jpkm1     ! search maximum in neighbourhood
         ikm1 = MAX(jk-1,1)
         DO_2D( 0, 0, 0, 0 )
            zbetup(ji,jj,jk) = MAX(  pbef(ji  ,jj  ,jk  ), paft(ji  ,jj  ,jk  ),   &
               &                     pbef(ji  ,jj  ,ikm1), pbef(ji  ,jj  ,jk+1),   &
               &                     paft(ji  ,jj  ,ikm1), paft(ji  ,jj  ,jk+1)  )
         END_2D
      END DO
      !                    ! large positive value (+zbig) inside land
      DO_3D( 0, 0, 0, 0, 1, jpk )
         pbef(ji,jj,jk) = pbef(ji,jj,jk) * tmask(ji,jj,jk) + zbig * ( 1.e0 - tmask(ji,jj,jk) )
         paft(ji,jj,jk) = paft(ji,jj,jk) * tmask(ji,jj,jk) + zbig * ( 1.e0 - tmask(ji,jj,jk) )
      END_3D
      !
      DO jk = 1, jpkm1     ! search minimum in neighbourhood
         ikm1 = MAX(jk-1,1)
         DO_2D( 0, 0, 0, 0 )
            zbetdo(ji,jj,jk) = MIN(  pbef(ji  ,jj  ,jk  ), paft(ji  ,jj  ,jk  ),   &
               &                     pbef(ji  ,jj  ,ikm1), pbef(ji  ,jj  ,jk+1),   &
               &                     paft(ji  ,jj  ,ikm1), paft(ji  ,jj  ,jk+1)  )
         END_2D
      END DO
      !                    ! restore masked values to zero
      DO_3D( 0, 0, 0, 0, 1, jpk )
         pbef(ji,jj,jk) = pbef(ji,jj,jk) * tmask(ji,jj,jk)
         paft(ji,jj,jk) = paft(ji,jj,jk) * tmask(ji,jj,jk)
      END_3D
      !
      ! Positive and negative part of fluxes and beta terms
      ! ---------------------------------------------------
      DO_3D( 0, 0, 0, 0, 1, jpkm1 )
         ! positive & negative part of the flux
         zpos = MAX( 0., pcc(ji  ,jj  ,jk+1) ) - MIN( 0., pcc(ji  ,jj  ,jk  ) )
         zneg = MAX( 0., pcc(ji  ,jj  ,jk  ) ) - MIN( 0., pcc(ji  ,jj  ,jk+1) )
         ! up & down beta terms
         zbt = e1e2t(ji,jj) * e3t(ji,jj,jk,Kmm) / pdt
         zbetup(ji,jj,jk) = ( zbetup(ji,jj,jk) - paft(ji,jj,jk) ) / (zpos+zrtrn) * zbt
         zbetdo(ji,jj,jk) = ( paft(ji,jj,jk) - zbetdo(ji,jj,jk) ) / (zneg+zrtrn) * zbt
      END_3D
      !
      ! monotonic flux in the k direction, i.e. pcc
      ! -------------------------------------------
      DO_3D( 0, 0, 0, 0, 2, jpkm1 )
         za = MIN( 1., zbetdo(ji,jj,jk), zbetup(ji,jj,jk-1) )
         zb = MIN( 1., zbetup(ji,jj,jk), zbetdo(ji,jj,jk-1) )
         zc = 0.5 * ( 1.e0 + SIGN( 1.0_wp, pcc(ji,jj,jk) ) )
         pcc(ji,jj,jk) = pcc(ji,jj,jk) * ( zc * za + ( 1.e0 - zc) * zb )
      END_3D
      !
   END SUBROUTINE nonosc_z

   !!======================================================================
END MODULE traadv_ubs
