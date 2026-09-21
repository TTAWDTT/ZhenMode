MODULE sshwzv
   !!==============================================================================
   !!                       ***  MODULE  sshwzv  ***
   !! Ocean dynamics : sea surface height and vertical velocity
   !!==============================================================================
   !! History :  3.1  !  2009-02  (G. Madec, M. Leclair)  Original code
   !!            3.3  !  2010-04  (M. Leclair, G. Madec)  modified LF-RA
   !!             -   !  2010-05  (K. Mogensen, A. Weaver, M. Martin, D. Lea)  Assimilation interface
   !!             -   !  2010-09  (D.Storkey and E.O'Dea)  bug fixes for BDY module
   !!            3.3  !  2011-10  (M. Leclair)  split former ssh_wzv routine and remove all vvl related work
   !!            4.0  !  2018-12  (A. Coward)  add mixed implicit/explicit advection
   !!            4.1  !  2019-08  (A. Coward, D. Storkey)  Rename ssh_nxt -> ssh_atf. Now only does time filtering.
   !!             -   !  2020-08  (S. Techene, G. Madec)  add here ssh initiatlisation
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   ssh_nxt       : after ssh
   !!   wzv           : compute a vertical velocity
   !!   wAimp         : partition vertical velocity for adaptive implicit option (ln_zad_Aimp=T)
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and tracers variables
   USE isf_oce        ! ice shelf
   USE dom_oce        ! ocean space and time domain variables
   USE domutl  , ONLY : lbnd_ij, in_hdom
   USE sbc_oce        ! surface boundary condition: ocean
   USE divhor         ! horizontal divergence
   USE phycst         ! physical constants
   USE bdy_oce , ONLY : ln_bdy, bdytmask   ! Open BounDarY
   USE bdydyn2d       ! bdy_ssh routine
   USE wet_dry        ! Wetting/Drying flux limiting
#if defined key_agrif
   USE agrif_oce
   USE agrif_oce_interp
#endif
   !
   USE iom
   USE in_out_manager ! I/O manager
   USE restart        ! only for lrst_oce
   USE prtctl         ! Print control
   USE lbclnk         ! ocean lateral boundary condition (or mpp link)
   USE lib_mpp        ! MPP library
   USE timing         ! Timing

   IMPLICIT NONE
   PRIVATE

   PUBLIC   ssh_nxt        ! called by stprk3.F90 (SWE) 
   PUBLIC   wzv            ! called by stp2d.F90, stprk3_stg.F90 and traadv.F90
   PUBLIC   wAimp          ! called by            stprk3_stg.F90 and traadv.F90

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE ssh_nxt( kt, Kbb, Kmm, pssh, Kaa )
      !!----------------------------------------------------------------------
      !!                ***  ROUTINE ssh_nxt  ***
      !!
      !! ** Purpose :   compute the after ssh (ssh(Kaa))
      !!
      !! ** Method  : - Using the incompressibility hypothesis, the ssh increment
      !!      is computed by integrating the horizontal divergence and multiply by
      !!      by the time step.
      !!
      !! ** action  :   ssh(:,:,Kaa), after sea surface height
      !!
      !! Reference  : Leclair, M., and G. Madec, 2009, Ocean Modelling.
      !!----------------------------------------------------------------------
      INTEGER                         , INTENT(in   ) ::   kt             ! time step
      INTEGER                         , INTENT(in   ) ::   Kbb, Kmm, Kaa  ! time level index
      REAL(wp), DIMENSION(jpi,jpj,jpt), INTENT(inout) ::   pssh           ! sea-surface height
      !
      INTEGER  ::   ji, jj, jk      ! dummy loop index
      REAL(wp) ::   zcoef   ! local scalar
      REAL(wp), DIMENSION(A2D(0)) ::   zhdiv   ! 2D workspace
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('ssh_nxt')
      !
      IF( kt == nit000 ) THEN
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) 'ssh_nxt : after sea surface height'
         IF(lwp) WRITE(numout,*) '~~~~~~~ '
      ENDIF
      !
      zcoef = 0.5_wp * r1_rho0
      !                                           !------------------------------!
      !                                           !   After Sea Surface Height   !
      !                                           !------------------------------!

      CALL div_hor( kt, Kbb, Kmm )                     ! Horizontal divergence
      !
      zhdiv(:,:) = 0._wp
      DO_3D( 0, 0, 0, 0, 1, jpkm1 )                                 ! Horizontal divergence of barotropic transports
        zhdiv(ji,jj) = zhdiv(ji,jj) + e3t(ji,jj,jk,Kmm) * hdiv(ji,jj,jk)
      END_3D
      !                                                ! Sea surface elevation time stepping
      ! In time-split case we need a first guess of the ssh after (using the baroclinic timestep) in order to
      ! compute the vertical velocity which can be used to compute the non-linear terms of the momentum equations.
      !
      DO_2D( 0, 0, 0, 0 )
         pssh(ji,jj,Kaa) = (  pssh(ji,jj,Kbb) - rDt * ( r1_rho0 * emp(ji,jj) + zhdiv(ji,jj) )  ) * ssmask(ji,jj)
      END_2D
      ! pssh must be defined everywhere
      CALL lbc_lnk( 'sshwzv', pssh(:,:,Kaa), 'T', 1.0_wp, ldfull=.TRUE. )
      !
#if defined key_agrif
      Kbb_a = Kbb   ;   Kmm_a = Kmm   ;   Krhs_a = Kaa
      CALL agrif_ssh( kt )
#endif
      !
      IF ( .NOT.ln_dynspg_ts ) THEN
         IF( ln_bdy ) THEN
            CALL bdy_ssh( pssh(:,:,Kaa) )              ! Duplicate sea level across open boundaries
         ENDIF
      ENDIF
      !
      !                                                ! Control print
      IF(sn_cfctl%l_prtctl)   CALL prt_ctl( tab2d_1=pssh(:,:,Kaa), clinfo1=' pssh(:,:,Kaa)  - : ', mask1=tmask )
      !
      IF( ln_timing )   CALL timing_stop('ssh_nxt')
      !
   END SUBROUTINE ssh_nxt


   SUBROUTINE wzv( kt, Kbb, Kmm, Kaa, pu, pv, pww, k_ind )
      !!
      INTEGER                         , INTENT(in   ) ::   kt             ! time step
      INTEGER                         , INTENT(in   ) ::   Kbb, Kmm, Kaa  ! time level indices
      INTEGER , OPTIONAL              , INTENT(in   ) ::   k_ind          ! indicator (np_transport or np_velocity)
      REAL(wp), DIMENSION(:,:,:)      , INTENT(in   ) ::   pu, pv         ! horizontal velocity at Kmm
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(inout) ::             pww  ! vertical velocity at Kmm
      !!
      CALL wzv_t( kt, Kbb, Kmm, Kaa, pu, pv, lbnd_ij(pu), pww, k_ind )
   END SUBROUTINE wzv


   SUBROUTINE wzv_t( kt, Kbb, Kmm, Kaa, pu, pv, ktuv, pww, k_ind )
      !!----------------------------------------------------------------------
      !!                ***  ROUTINE wzv  ***
      !!
      !! ** Purpose :   compute the now vertical velocity
      !!
      !! ** Method  : - Using the incompressibility hypothesis, the vertical
      !!      velocity is computed by integrating the horizontal divergence
      !!      from the bottom to the surface minus the scale factor evolution.
      !!        The boundary conditions are w=0 at the bottom (no flux) and.
      !!
      !! ** action  :   pww      : now vertical velocity
      !!
      !! Reference  : Leclair, M., and G. Madec, 2009, Ocean Modelling.
      !!----------------------------------------------------------------------
      INTEGER,  DIMENSION(2)             , INTENT(in   ) ::   ktuv
      INTEGER                            , INTENT(in   ) ::   kt             ! time step
      INTEGER                            , INTENT(in   ) ::   Kbb, Kmm, Kaa  ! time level indices
      INTEGER , OPTIONAL                 , INTENT(in   ) ::   k_ind          ! indicator (np_transport or np_velocity)
      REAL(wp), DIMENSION(AB2D(ktuv),JPK), INTENT(in   ) ::   pu, pv         ! horizontal velocity at Kmm
      REAL(wp), DIMENSION(jpi,jpj,jpk)   , INTENT(inout) ::             pww  ! vertical velocity at Kmm
      !
      INTEGER  ::   ji, jj, jk   ! dummy loop indices
      REAL(wp), DIMENSION(T2D(1),jpk) ::   ze3div
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('wzv')
      !
      IF( kt == nit000 ) THEN
         IF( .NOT. l_istiled .OR. ntile == 1 )  THEN                       ! Do only on the first tile
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) 'wzv : now vertical velocity '
            IF(lwp) WRITE(numout,*) '~~~~~ '
         ENDIF
         !
         DO_2D( 1, 1, 1, 1 )
            pww(ji,jj,jpk) = 0._wp                  ! bottom boundary condition: w=0 (set once for all)
         END_2D
         !                                   ! needed over the halos for the output (ww+wi) in diawri.F90
      ENDIF
      !
      IF( .NOT. PRESENT( k_ind ) ) THEN
         CALL div_hor( kt, Kbb, Kmm, pu, pv, ze3div )
      ELSE
         CALL div_hor( kt, Kbb, Kmm, pu, pv, ze3div, k_ind )
      ENDIF
      !                                           !------------------------------!
      !                                           !     Now Vertical Velocity    !
      !                                           !------------------------------!
      !
      !                                               !=================================!
      IF( lk_linssh )   THEN                          !==  linear free surface cases  ==!
         !                                            !=================================!
         DO_3DS( 1, 1, 1, 1, jpkm1, 1, -1 )     ! integrate from the bottom the hor. divergence
            pww(ji,jj,jk) = pww(ji,jj,jk+1) - ze3div(ji,jj,jk)
         END_3D
         !                                            !==========================================!
      ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!   ('key_qco')
         !                                            !==========================================!
         DO_3DS( 1, 1, 1, 1, jpkm1, 1, -1 )     ! integrate from the bottom the hor. divergence
            !                                                              ! NB: [e3t[a] -e3t[b] ]=e3t_0*[r3t[a]-r3t[b]]
            pww(ji,jj,jk) = pww(ji,jj,jk+1) - (  ze3div(ji,jj,jk)                             &
               &                               + r1_Dt * e3t_0(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )  ) * tmask(ji,jj,jk)
         END_3D
      ENDIF

      IF( ln_bdy ) THEN
         DO_3D( 1, 1, 1, 1, 1, jpkm1 )
            pww(ji,jj,jk) = pww(ji,jj,jk) * bdytmask(ji,jj)
         END_3D
      ENDIF
      !
#if defined key_agrif
      IF( .NOT. AGRIF_Root() ) THEN
         !
         ! Mask vertical velocity at first/last columns/row
         ! inside computational domain (cosmetic)
         DO jk = 1, jpkm1
            IF( lk_west ) THEN                             ! --- West --- !
               DO ji = mi0(2+nn_hls,nn_hls), mi1(2+nn_hls,nn_hls)
                  DO jj = 2, jpj-1
                     IF( in_hdom(ji, jj, khls=1) ) pww(ji,jj,jk) = 0._wp
                  END DO
               END DO
            ENDIF
            IF( lk_east ) THEN                             ! --- East --- !
               DO ji = mi0(jpiglo-nn_hls-1,nn_hls), mi1(jpiglo-nn_hls-1,nn_hls)
                  DO jj = 2, jpj-1
                     IF( in_hdom(ji, jj, khls=1) ) pww(ji,jj,jk) = 0._wp
                  END DO
               END DO
            ENDIF
            IF( lk_south ) THEN                            ! --- South --- !
               DO jj = mj0(2+nn_hls,nn_hls), mj1(2+nn_hls,nn_hls)
                  DO ji = 2, jpi-1
                     IF( in_hdom(ji, jj, khls=1) ) pww(ji,jj,jk) = 0._wp
                  END DO
               END DO
            ENDIF
            IF( lk_north ) THEN                            ! --- North --- !
               DO jj = mj0(jpjglo-nn_hls-1,nn_hls), mj1(jpjglo-nn_hls-1,nn_hls)
                  DO ji = 2, jpi-1
                     IF( in_hdom(ji, jj, khls=1) ) pww(ji,jj,jk) = 0._wp
                  END DO
               END DO
            ENDIF
            !
         END DO
         !
      ENDIF
#endif
      !
      IF( ln_timing )   CALL timing_stop('wzv')
      !
   END SUBROUTINE wzv_t


   SUBROUTINE wAimp( kt, Kmm, puu, pvv, pww, pwi, k_ind, pCu_min_v, pCu_max_v, pCu_max_h, ld_diag )
      !!
      INTEGER                         , INTENT(in   ) ::   kt             ! time step
      INTEGER                         , INTENT(in   ) ::   Kmm            ! time level index
      INTEGER                         , INTENT(in   ) ::   k_ind          ! indicator (np_transport or np_velocity)
      REAL(wp), DIMENSION(:,:,:)      , INTENT(in   ) ::   puu, pvv       !  horizontal velocity at Kmm
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(inout) ::   pww            !  vertical velocity at Kmm (explicit part)
      REAL(wp), DIMENSION(:,:,:)      , INTENT(inout) ::   pwi            !  vertical velocity at Kmm (implicit part)
      REAL(wp)                        , INTENT(in)    ::   pCu_min_v      ! minimum vertical CFL above which starts the partioning
      REAL(wp)                        , INTENT(in)    ::   pCu_max_v      ! maximum vertical CFL allowed
      REAL(wp)                        , INTENT(in)    ::   pCu_max_h      ! maximum horizontal CFL allowed
      LOGICAL, OPTIONAL               , INTENT(in   ) ::   ld_diag        !  =true : write implicit outputs
      !!
      CALL wAimp_t( kt, Kmm, puu, pvv, lbnd_ij(puu), pww, pwi, lbnd_ij(pwi), k_ind, pCu_min_v, pCu_max_v, pCu_max_h, ld_diag )
   END SUBROUTINE wAimp


   SUBROUTINE wAimp_t( kt, Kmm, puu, pvv, ktuv, pww, pwi, ktwi, k_ind, pCu_min_v, pCu_max_v, pCu_max_h, ld_diag )
      !!----------------------------------------------------------------------
      !!                ***  ROUTINE wAimp  ***
      !!
      !! ** Purpose :   compute the Courant number and partition vertical velocity
      !!                if a proportion needs to be treated implicitly
      !!
      !! ** Method  : -
      !!
      !! ** action  :   ww      : now vertical velocity (to be handled explicitly)
      !!            :   wi      : now vertical velocity (for implicit treatment)
      !!
      !! Reference  : Shchepetkin, A. F. (2015): An adaptive, Courant-number-dependent
      !!              implicit scheme for vertical advection in oceanic modeling.
      !!              Ocean Modelling, 91, 38-69.
      !!
      !!              Wicker, L. J. and W. C. Skamarock (2020): An Implicit-Explicit
      !!              Vertical Transport Scheme for Convection-Allowing Models.
      !!              Monthly Weather Review, 148:9, 3893-S3910.
      !!              https://doi.org/10.1175/MWR-D-20-0055.1
      !!----------------------------------------------------------------------
      INTEGER,  DIMENSION(2)             , INTENT(in   ) ::   ktuv, ktwi
      INTEGER                            , INTENT(in   ) ::   kt             ! time step
      INTEGER                            , INTENT(in   ) ::   Kmm            ! time level index
      INTEGER                            , INTENT(in   ) ::   k_ind          ! indicator (np_transport or np_velocity)
      REAL(wp), DIMENSION(AB2D(ktuv),JPK), INTENT(in   ) ::   puu, pvv       ! horizontal velocity at Kmm
      REAL(wp), DIMENSION(jpi,jpj,jpk)   , INTENT(inout) ::   pww            ! vertical velocity at Kmm (explicit part)
      REAL(wp), DIMENSION(AB2D(ktwi),JPK), INTENT(inout) ::   pwi            ! vertical velocity at Kmm (implicit part)
      REAL(wp)                           , INTENT(in)    ::   pCu_min_v      ! minimum vertical CFL above which starts the partioning
      REAL(wp)                           , INTENT(in)    ::   pCu_max_v      ! maximum vertical CFL allowed
      REAL(wp)                           , INTENT(in)    ::   pCu_max_h      ! maximum horizontal CFL allowed
      LOGICAL, OPTIONAL                  , INTENT(in   ) ::   ld_diag        !  =true : write implicit outputs
      !
      INTEGER  ::   ji, jj, jk   ! dummy loop indices
      REAL(wp)             ::   zcff, z1_e3t, z1_e3w, zdt, zCu_h, zCu_v   !  local scalars
      REAL(wp)             ::   zCu_min, zCu_max, zCu_cut, zr_Cu_max_h    !  local scalar
      LOGICAL  :: ll_diag
      CHARACTER(LEN=10) :: clmname
      REAL(wp), DIMENSION(:,:        ), ALLOCATABLE :: zdiag2d
      REAL(wp), DIMENSION(:,:,:      ), ALLOCATABLE :: zdiag3d
      REAL(wp), DIMENSION(T2D(nn_hls))              :: z2d
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('wAimp')
      !
      IF( kt == nit000 ) THEN
         IF( .NOT. l_istiled .OR. ntile == 1 )  THEN                       ! Do only on the first tile
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) 'wAimp : Courant number-based partitioning of now vertical velocity '
            IF(lwp) WRITE(numout,*) '~~~~~ '
            IF(lwp) WRITE(numout,'(3(a,F10.4,1x))') 'Partitioning parameters: Cu_min_v= ', pCu_min_v, &
            &                                       'Cu_max_v= ', pCu_max_v,  'Cu_max_h= ', pCu_max_h
         ENDIF
      ENDIF
      !
      ll_diag = .FALSE.
      IF( PRESENT(ld_diag) ) ll_diag = ld_diag 
      ! Calculate Courant numbers
      !
      zdt = rn_Dt                    ! 3rd stage timestep
      zr_Cu_max_h = 1._wp/pCu_max_h
      !
      ! Sort of horizontal Courant number:
      ! JC: Is it still worth saving into a 3d array ? I don't believe.
      SELECT CASE ( k_ind )
      CASE ( np_velocity )
         DO_3D( 1, 1, 1, 1, 1, jpkm1 )
            z1_e3t = 1._wp / e3t(ji,jj,jk,Kmm)
            Cu_adv(ji,jj,jk) =   zdt *                                                      &
               &  ( ( MAX( e2u(ji  ,jj)*e3u(ji  ,jj,jk,Kmm)*puu(ji  ,jj,jk), 0._wp ) -   &
               &      MIN( e2u(ji-1,jj)*e3u(ji-1,jj,jk,Kmm)*puu(ji-1,jj,jk), 0._wp ) )   &
               &  + ( MAX( e1v(ji,jj  )*e3v(ji,jj  ,jk,Kmm)*pvv(ji,jj  ,jk), 0._wp ) -   &
               &      MIN( e1v(ji,jj-1)*e3v(ji,jj-1,jk,Kmm)*pvv(ji,jj-1,jk), 0._wp ) )   &
               &                             ) * z1_e3t * r1_e1e2t(ji,jj)
         END_3D
      CASE ( np_transport )
         DO_3D( 1, 1, 1, 1, 1, jpkm1 )
            z1_e3t = 1._wp / e3t(ji,jj,jk,Kmm)
            Cu_adv(ji,jj,jk) =   zdt *                                                      &
               &  ( ( MAX( puu(ji  ,jj,jk), 0._wp ) -   &
               &      MIN( puu(ji-1,jj,jk), 0._wp ) )   &
               &  + ( MAX( pvv(ji,jj  ,jk), 0._wp ) -   &
               &      MIN( pvv(ji,jj-1,jk), 0._wp ) )   &
               &  ) * z1_e3t * r1_e1e2t(ji,jj)
         END_3D
      END SELECT
      !
      ! JC: Warning: this is the horizontal Courant number this time
      ! not the total as in previous versions of the scheme.
      !
      IF( ll_diag .AND. iom_use("Aimp_Cmx_h") ) THEN
         ALLOCATE( zdiag2d(T2D(0)) )
         zdiag2d(:,:) = MAXVAL(Cu_adv(T2D(0),:), DIM=3)
         zdiag2d(:,:) = Cu_adv(T2D(0),10)
         CALL iom_put( 'Aimp_Cmx_h', zdiag2d(:,:) )
         DEALLOCATE( zdiag2d )
      ENDIF
      !
      z2d(:,:) = 0._wp
      DO_3DS( 1, 1, 1, 1, jpkm1, 2, -1 )
         !
         zcff = z2d(ji,jj)
         IF ( pww(ji,jj,jk) > 0._wp ) THEN
            zCu_h =  Cu_adv(ji,jj,jk  )
         ELSE
            zCu_h =  Cu_adv(ji,jj,jk-1)
         ENDIF
         ! Vertical Courant Number:
         z1_e3w = 1._wp / e3w(ji,jj,jk,Kmm)
         zCu_v = zdt * z1_e3w * ABS (pww(ji,jj,jk))
         z2d(ji,jj) = MAX( zCu_v, zcff )
         !
         zCu_min = pCu_min_v * (1._wp - zCu_h * zr_Cu_max_h)
         zCu_max = pCu_max_v * (1._wp - zCu_h * zr_Cu_max_h)
         zCu_cut = 2._wp * zCu_max - zCu_min
         !
         IF( zCu_v <= zCu_min ) THEN            !<-- Fully explicit
            zcff = 0._wp
         ELSEIF( zCu_v < zCu_cut ) THEN         !<-- Mixed explicit
            zcff = 1.0_wp / ( zCu_v - zCu_min )**2
            zcff = 1.0_wp / ( 1.0_wp + 4._wp * zCu_max * (zCu_max - zCu_min) * zcff )
         ELSE                                   !<-- Mostly implicit
            zcff = ( zCu_v - zCu_max ) / zCu_v
         ENDIF
         zcff = MIN(1._wp, zcff)
         zcff = MAX(0._wp, zcff)
         !
         ! Split vertical velocity:
         pwi(ji,jj,jk) =           zcff   * pww(ji,jj,jk)
         pww(ji,jj,jk) = ( 1._wp - zcff ) * pww(ji,jj,jk)
         !
         Cu_adv(ji,jj,jk) = zcff               ! Reuse array to output coefficient below and in stp_ctl
      END_3D
      Cu_adv(T2D(1),1) = 0._wp
      !
      IF( ll_diag ) THEN
         IF( iom_use("wimp") ) THEN
            ! Fix for tiling: force data to be sent to XIOS by copying pwi to a temporary array.
            ! The problem: iom_put will only send data on the last tile since pwi has size (jpi,jpj,jpk), and pwi will
            ! be overwritten by the 2nd call to wAimp. Therefore when pwi is actually sent to XIOS, all tiles except
            ! the last will have data equivalent to k_ind == np_transport (rather than k_ind == np_velocity).
            ALLOCATE( zdiag3d(T2D(0),jpk) )
            zdiag3d(:,:,1:jpk) = pwi(T2D(0),1:jpk)
            CALL iom_put( "wimp", zdiag3d(:,:,:) )
            DEALLOCATE( zdiag3d )
         ENDIF
         IF( iom_use("wi_cff") ) THEN
            ALLOCATE( zdiag3d(T2D(0),jpk) )
            zdiag3d(:,:,1:jpk) = Cu_adv(T2D(0),1:jpk)
            CALL iom_put( "wi_cff", zdiag3d(:,:,:) )
            DEALLOCATE( zdiag3d )
         ENDIF
         CALL iom_put( "Aimp_Cmx_v", z2d(:,:) )      ! o/p column maximum vertical Courant number
         IF( iom_use("Aimp_loc") ) THEN
            ALLOCATE( zdiag2d(T2D(0)) )
            WHERE( SUM( Cu_adv(T2D(0),:), DIM=3 ) > rsmall ) ; zdiag2d(:,:) = 1._wp
            ELSEWHERE                                        ; zdiag2d(:,:) = 0._wp
            ENDWHERE
            CALL iom_put( "Aimp_loc", zdiag2d(:,:) )      ! o/p active Aimp locations
            DEALLOCATE( zdiag2d )
         ENDIF
      ENDIF
      !
      IF( ln_timing )   CALL timing_stop('wAimp')
      !
   END SUBROUTINE wAimp_t
   !!======================================================================
END MODULE sshwzv
