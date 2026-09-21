MODULE traadv_mus
   !!======================================================================
   !!                       ***  MODULE  traadv_mus  ***
   !! Ocean  tracers:  horizontal & vertical advective trend
   !!======================================================================
   !! History :       !  2000-06  (A.Estublier)  for passive tracers
   !!                 !  2001-08  (E.Durand, G.Madec)  adapted for T & S
   !!   NEMO     1.0  !  2002-06  (G. Madec)  F90: Free form and module
   !!            3.2  !  2010-05  (C. Ethe, G. Madec)  merge TRC-TRA + switch from velocity to transport
   !!            3.4  !  2012-06  (P. Oddo, M. Vichi) include the upstream where needed
   !!            3.7  !  2015-09  (G. Madec) add the ice-shelf cavities boundary condition
   !!            4.5  !  2022-06  (S. Techene, G, Madec) refactorization to reduce local memory usage
   !!            5.x  !  2026-03  (S. Griffies, G. Madec)  thickness weighted tracer tendency 
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   tra_adv_mus   : update the tracer trend with the horizontal
   !!                   and vertical advection trends using MUSCL scheme
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and active tracers
   USE trc_oce        ! share passive tracers/Ocean variables
   USE dom_oce        ! ocean space and time domain
   USE trd_oce        ! trends: ocean variables
   USE trdtra         ! tracers trends manager
   USE sbcrnf         ! river runoffs
   USE diaptr         ! poleward transport diagnostics
   USE diaar5         ! AR5 diagnostics
   USE domutl, ONLY : lbnd_ij

   !
   USE iom            ! XIOS library
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! distribued memory computing
   USE lbclnk         ! ocean lateral boundary condition (or mpp link)
   USE lib_fortran    ! Fortran utilities (allows no signed zero when 'key_nosignedzero' defined)

   IMPLICIT NONE
   PRIVATE

   PUBLIC   tra_adv_mus        ! routine called by traadv.F90


   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:)   ::   upsmsk   !: mixed upstream/centered scheme near some straits
   !                                                           !  and in closed seas (orca 2 and 1 configurations)
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   xind     !: mixed upstream/centered index

   LOGICAL  ::   l_trd   ! flag to compute trends
   LOGICAL  ::   l_ptr   ! flag to compute poleward transport
   LOGICAL  ::   l_hst   ! flag to compute heat/salt transport
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_thres_tra_mus = 0.85_wp  ! starting Courant number threshold for adaptive implicit vertical advection
   REAL(wp), PUBLIC, PARAMETER :: pp_stb_cstra_tra_mus = 1.00_wp  ! stability constraint

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE tra_adv_mus( kt, kit000, cdtype, pdt, pU, pV, pW,             &
      &                    Kbb, Kmm, pt, kjpt, Krhs, ld_msc_ups )
      !!
      INTEGER                                  , INTENT(in   ) ::   kt              ! ocean time-step index
      INTEGER                                  , INTENT(in   ) ::   Kbb, Kmm, Krhs  ! ocean time level indices
      INTEGER                                  , INTENT(in   ) ::   kit000          ! first time step index
      CHARACTER(len=3)                         , INTENT(in   ) ::   cdtype          ! =TRA or TRC (tracer indicator)
      INTEGER                                  , INTENT(in   ) ::   kjpt            ! number of tracers
      LOGICAL                                  , INTENT(in   ) ::   ld_msc_ups      ! use upstream scheme within muscl
      REAL(wp)                                 , INTENT(in   ) ::   pdt             ! tracer time-step
      REAL(wp), DIMENSION(:,:,:               ), INTENT(in   ) ::   pU, pV, pW      ! 3 ocean volume flux components
      REAL(wp), DIMENSION(jpi,jpj,jpk,kjpt,jpt), INTENT(inout) ::   pt              ! tracers and RHS of tracer equation
      !!
      CALL tra_adv_mus_t( kt, kit000, cdtype, pdt, pU, pV, pW, lbnd_ij(pU),    &
        &                 Kbb, Kmm, pt, kjpt, Krhs, ld_msc_ups                  )
   END SUBROUTINE tra_adv_mus

   SUBROUTINE tra_adv_mus_t( kt, kit000, cdtype, pdt, pU, pV, pW, ktpuvw,  &
      &                      Kbb, Kmm, pt, kjpt, Krhs, ld_msc_ups           )
      !!----------------------------------------------------------------------
      !!                    ***  ROUTINE tra_adv_mus  ***
      !!
      !! ** Purpose :   Compute the now trend due to total advection of tracers
      !!              using a MUSCL scheme (Monotone Upstream-centered Scheme for
      !!              Conservation Laws) and add it to the general tracer trend.
      !!
      !! ** Method  : MUSCL scheme plus centered scheme at ocean boundaries
      !!              ld_msc_ups=T :
      !!
      !! ** Action : - update pt(:,:,:,:,Krhs)  with the now advective tracer trends
      !!             - send trends to trdtra module for further diagnostcs (l_trdtra=T)
      !!             - poleward advective heat and salt transport (ln_diaptr=T)
      !!
      !! References : Estubier, A., and M. Levy, Notes Techn. Pole de Modelisation
      !!              IPSL, Sept. 2000 (http://www.lodyc.jussieu.fr/opa)
      !!----------------------------------------------------------------------
      INTEGER,  DIMENSION(2)                   , INTENT(in   ) ::   ktpuvw
      INTEGER                                  , INTENT(in   ) ::   kt              ! ocean time-step index
      INTEGER                                  , INTENT(in   ) ::   Kbb, Kmm, Krhs  ! ocean time level indices
      INTEGER                                  , INTENT(in   ) ::   kit000          ! first time step index
      CHARACTER(len=3)                         , INTENT(in   ) ::   cdtype          ! =TRA or TRC (tracer indicator)
      INTEGER                                  , INTENT(in   ) ::   kjpt            ! number of tracers
      LOGICAL                                  , INTENT(in   ) ::   ld_msc_ups      ! use upstream scheme within muscl
      REAL(wp)                                 , INTENT(in   ) ::   pdt             ! tracer time-step
      REAL(wp), DIMENSION(AB2D(ktpuvw),JPK    ), INTENT(in   ) ::   pU, pV, pW      ! 3 ocean volume flux components
      REAL(wp), DIMENSION(jpi,jpj,jpk,kjpt,jpt), INTENT(inout) ::   pt              ! tracers and RHS of tracer equation
      !
      INTEGER  ::   ji, jj, jk, jn   ! dummy loop indices
      INTEGER  ::   ierr             ! local integer
      INTEGER  ::   ik
      REAL(wp) ::   zu, z0u, zzslpx, zzwx, zw , zalpha   ! local scalars
      REAL(wp) ::   zv, z0v, zzslpy, zzwy, z0w           !   -      -
      REAL(wp) ::   zdzt_kp2, zslpz_kp1, zfW_kp1
      REAL(wp), DIMENSION(T2D(2)) ::   zdxt, zslpx, zwx  ! 2D workspace
      REAL(wp), DIMENSION(T2D(2)) ::   zdyt, zslpy, zwy  ! -      -
      !!----------------------------------------------------------------------
      !
      IF( .NOT. l_istiled .OR. ntile == 1 )  THEN                       ! Do only on the first tile
         IF( kt == kit000 )  THEN
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) 'tra_adv : MUSCL advection scheme on ', cdtype
            IF(lwp) WRITE(numout,*) '        : mixed up-stream           ', ld_msc_ups
            IF(lwp) WRITE(numout,*) '~~~~~~~'
            IF(lwp) WRITE(numout,*)
            !
            ! Upstream / MUSCL scheme indicator
            !
            ALLOCATE( xind(jpi,jpj,jpk), STAT=ierr )
            xind(:,:,:) = 1._wp              ! set equal to 1 where up-stream is not needed
            IF( ld_msc_ups ) THEN            ! define the upstream indicator (if asked)
               DO jk = 1, jpkm1
                  xind(:,:,jk) = 1._wp                              &                 ! =>1 where up-stream is not needed
                     &                -  rnfmsk(:,:) * rnfmsk_z(jk) * tmask(:,:,jk)   ! =>0 near runoff mouths (& closed sea outflows) 
               END DO
            ENDIF
            !
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
      DO jn = 1, kjpt            !==  loop over the tracers  ==!
         !
         DO jk = 1, jpkm1
            !                          !* Horizontal advective fluxes
            !
            !                                !-- first guess of the slopes
            DO_2D( 2, 1, 2, 1 )
               zdxt(ji,jj) = ( pt(ji+1,jj  ,jk,jn,Kbb) - pt(ji,jj,jk,jn,Kbb) ) * umask(ji,jj,jk)
               zdyt(ji,jj) = ( pt(ji  ,jj+1,jk,jn,Kbb) - pt(ji,jj,jk,jn,Kbb) ) * vmask(ji,jj,jk)
            END_2D
            !                                !-- Slopes of tracer
            DO_2D( 1, 1, 1, 1 )
               !                                 ! 1/2 Slopes at T-point (set to 0 if adjectent slopes are of opposite sign)
               zzslpx =                          ( zdxt(ji,jj) + zdxt(ji-1,jj  ) )   &
                  &   * ( 0.25_wp + SIGN( 0.25_wp, zdxt(ji,jj) * zdxt(ji-1,jj  ) ) )
               zzslpy =                          ( zdyt(ji,jj) + zdyt(ji  ,jj-1) )   &
                  &   * ( 0.25_wp + SIGN( 0.25_wp, zdyt(ji,jj) * zdyt(ji  ,jj-1) ) )
               !                                 ! Slopes limitation
               zslpx(ji,jj) = SIGN( 1.0_wp, zzslpx ) * MIN(       ABS( zzslpx         ),   &
                  &                                         2._wp*ABS( zdxt (ji-1,jj) ),   &
                  &                                         2._wp*ABS( zdxt (ji  ,jj) )    )
               zslpy(ji,jj) = SIGN( 1.0_wp, zzslpy ) * MIN(       ABS( zzslpy         ),   &
                  &                                         2._wp*ABS( zdyt (ji,jj-1) ),   &
                  &                                         2._wp*ABS( zdyt (ji,jj  ) )    )
            END_2D
!!gm + !!st ticket ? comparaison pommes et carrottes ABS(zzslpx) et 2._wp*ABS( zdxt (ji-1,jj) )
            !
            DO_2D( 1, 0, 1, 0 )              !-- MUSCL horizontal advective fluxes
               z0u = SIGN( 0.5_wp, pU(ji,jj,jk) )
               zalpha = 0.5_wp - z0u
               zu  = z0u - 0.5_wp * pU(ji,jj,jk) * pdt * r1_e1e2u(ji,jj) / e3u(ji,jj,jk,Kmm)
               zzwx = pt(ji+1,jj,jk,jn,Kbb) + xind(ji,jj,jk) * zu * zslpx(ji+1,jj)
               zzwy = pt(ji  ,jj,jk,jn,Kbb) + xind(ji,jj,jk) * zu * zslpx(ji  ,jj)
               zwx(ji,jj) = pU(ji,jj,jk) * ( zalpha * zzwx + (1._wp-zalpha) * zzwy )
               !
               z0v = SIGN( 0.5_wp, pV(ji,jj,jk) )
               zalpha = 0.5_wp - z0v
               zv  = z0v - 0.5_wp * pV(ji,jj,jk) * pdt * r1_e1e2v(ji,jj) / e3v(ji,jj,jk,Kmm)
               zzwx = pt(ji,jj+1,jk,jn,Kbb) + xind(ji,jj,jk) * zv * zslpy(ji,jj+1)
               zzwy = pt(ji,jj  ,jk,jn,Kbb) + xind(ji,jj,jk) * zv * zslpy(ji,jj  )
               zwy(ji,jj) = pV(ji,jj,jk) * ( zalpha * zzwx + (1._wp-zalpha) * zzwy )
            END_2D
            !
            DO_2D( 0, 0, 0, 0 )              !-- Tracer advective trend
               pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) - ( ( zwx(ji,jj) - zwx(ji-1,jj  ) )       &   ! ad () for NP repro
               &                                             + ( zwy(ji,jj) - zwy(ji  ,jj-1) ) )  * r1_e1e2t(ji,jj) 
            END_2D
            !                                 ! "Poleward" heat and salt transports
            IF( l_ptr )  CALL dia_ptr_hst( jn, 'adv', zwy(:,:) )
            !                                 !  heat transport
            IF( l_hst )  CALL dia_ar5_hst( jn, 'adv', zwx(:,:), zwy(:,:), ldfin=(jk == jpkm1) )
         END DO
!!gm + !!st to be done with the whole rewritting of trd
!!          trd routine arguments MUST be changed adding jk and zwx, zwy in 2D
!!
!!         !                                ! trend diagnostics
!!         IF( l_trd )  THEN
!!            CALL trd_tra( kt, Kmm, Krhs, cdtype, jn, jk, jptra_xad, zwx(:,:), pU, pt(:,:,:,jn,Kbb) )
!!            CALL trd_tra( kt, Kmm, Krhs, cdtype, jn, jk, jptra_yad, zwy(:,:), pV, pt(:,:,:,jn,Kbb) )
!!         END IF
         !
         !                          !* Vertical advective fluxes
         !
#define zdzt_kp1 zdxt 
#define zslpz zslpx
#define zfW zwx 

         zfW     (T2D(0)) = 0._wp    ! anciennement zwx at jk = 1
         !                          ! anciennement zwx at jk = 2
         DO_2D( 0, 0, 0, 0 )
            zdzt_kp1(ji,jj) = tmask(ji,jj,2) * ( pt(ji,jj,1,jn,Kbb) - pt(ji,jj,2,jn,Kbb) )
         END_2D
         zslpz   (T2D(0)) = 0._wp    ! anciennement zslpx at jk = 1
         !
         IF( lk_linssh ) THEN                !-- linear ssh : non zero top values
            DO_2D( 0, 0, 0, 0 )                       ! at the ocean surface
               zfW(ji,jj) = pW(ji,jj,1) * pt(ji,jj,1,jn,Kbb)    ! surface flux
            END_2D
            IF( ln_isfcav ) THEN                      ! ice-shelf cavities (top of the ocean)
               DO_2D( 0, 0, 0, 0 )                              ! update pt(Krhs) under the ice-shelf  
                  ik = mikt(ji,jj)                              ! the flux at ik-1 is zero ( inside ice-shelf )
                  IF( ik > 1 ) THEN
                     pt(ji,jj,ik,jn,Krhs) =  pt(ji,jj,ik,jn,Krhs) - pW(ji,jj,ik) * pt(ji,jj,ik,jn,Kbb) * r1_e1e2t(ji,jj) 
                  ENDIF
               END_2D              
            ENDIF
         ENDIF
         !
         ! wmask usage for computing zw and zwk is needed in isf case and linear ssh
         ! 
         !
         DO jk = 1, jpkm1
            IF( jk < jpkm1 ) THEN
               DO_2D( 0, 0, 0, 0 )
                  !                          !-- Slopes of tracer
                  !                                   ! masked vertical gradient at jk+2
                  zdzt_kp2 = ( pt(ji,jj,jk+1,jn,Kbb) - pt(ji,jj,jk+2,jn,Kbb) ) * wmask(ji,jj,jk+2)
                  !                                   ! vertical slope at jk+1
                  zslpz_kp1 =                           ( zdzt_kp1(ji,jj) + zdzt_kp2 )  &
                  &         * (  0.25_wp + SIGN( 0.25_wp, zdzt_kp1(ji,jj) * zdzt_kp2 )  )
                  !                                   ! slope limitation
                  zslpz_kp1 = SIGN( 1.0_wp, zslpz_kp1 ) * MIN(    ABS( zslpz_kp1 ),   &
                  &                                            2.*ABS( zdzt_kp2 ),   &
                  &                                            2.*ABS( zdzt_kp1(ji,jj) )  )
                  !                          !-- vertical advective flux at jk+1
                  !                          !    caution: zfW_kp1 is masked for ice-shelf cavities
                  !                          !    since top fluxes already added to pt(Krhs) before the vertical loop  
                  z0w = SIGN( 0.5_wp, pW(ji,jj,jk+1) )
                  zalpha = 0.5_wp + z0w
                  zw  = z0w - 0.5_wp * pW(ji,jj,jk+1) * pdt * r1_e1e2t(ji,jj) / e3w(ji,jj,jk+1,Kmm)
                  zzwx = pt(ji,jj,jk+1,jn,Kbb) + xind(ji,jj,jk) * zw * zslpz_kp1
                  zzwy = pt(ji,jj,jk  ,jn,Kbb) + xind(ji,jj,jk) * zw * zslpz(ji,jj)
                  zfW_kp1 = pW(ji,jj,jk+1) * ( zalpha * zzwx + (1.-zalpha) * zzwy ) * wmask(ji,jj,jk+1)
                  !                         !-- vertical advective trend at jk
                  pt(ji,jj,jk,jn,Krhs) =  pt(ji,jj,jk,jn,Krhs) - ( zfW(ji,jj) - zfW_kp1 ) * r1_e1e2t(ji,jj)
                  !                                   ! updates for next level
                  zdzt_kp1(ji,jj) = zdzt_kp2
                  zslpz   (ji,jj) = zslpz_kp1
                  zfW     (ji,jj) = zfW_kp1
               END_2D
            ELSE
               DO_2D( 0, 0, 0, 0 )          !-- vertical advective trend at jpkm1
                  pt(ji,jj,jk,jn,Krhs) =  pt(ji,jj,jk,jn,Krhs) - zfW(ji,jj) * r1_e1e2t(ji,jj) 
               END_2D
            ENDIF
	 END DO                  ! end of jk loop
         !
!!gm + !!st idem see above
!!         !                                ! send trends for diagnostic
!!         IF( l_trd )  CALL trd_tra( kt, Kmm, Krhs, cdtype, jn, jptra_zad, zwx, pW, pt(:,:,:,jn,Kbb) )
         !
      END DO                     ! end of tracer loop
      !
   END SUBROUTINE tra_adv_mus_t

   !!======================================================================
END MODULE traadv_mus
