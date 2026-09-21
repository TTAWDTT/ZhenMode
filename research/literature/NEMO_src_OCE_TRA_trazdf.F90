MODULE trazdf
   !!==============================================================================
   !!                 ***  MODULE  trazdf  ***
   !! Ocean active tracers:  vertical component of the tracer mixing trend
   !!==============================================================================
   !! History :  1.0  !  2005-11  (G. Madec)  Original code
   !!            3.0  !  2008-01  (C. Ethe, G. Madec)  merge TRC-TRA
   !!            4.0  !  2017-06  (G. Madec)  remove explict time-stepping option
   !!            4.5  !  2022-06  (G. Madec)  refactoring to reduce memory usage (j-k-i loops)
   !!            5.x  !  2026-03  (S. Griffies, G. Madec)  thickness weighted tracer tendency 
   !!                 !                                  + total & individual zdf tendencies
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   tra_zdf       : Update the tracer trend with the vertical diffusion
   !!   tra_zdf_imp   : inversion of the implicit matrix
   !!   tra_zdf_trd   : diagnose the vertical implicit e3t*trend (total and split terms)
   !!      zdf_trd_imp: diagnose vertical implicit diffusive e3t*trend (called from tra_zdf_trd)
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and tracers variables
   USE dom_oce        ! ocean space and time domain variables
   USE phycst         ! physical constant
   USE zdf_oce        ! ocean vertical physics variables
   USE zdfevd   , ONLY: avt_evd   ! enhanced vertical diffusion coefficient
   USE zdfmfc         ! Mass FLux Convection
   USE sbc_oce        ! surface boundary condition: ocean
   USE ldftra         ! lateral diffusion: eddy diffusivity
   USE ldfslp         ! lateral diffusion: iso-neutral slope
   USE trd_oce        ! trends: ocean variables
   USE trdtra         ! trends: tracer trend manager
   USE eosbn2   , ONLY: ln_SEOS, rn_b0
# if defined key_top
   USE trcldf   , ONLY: ln_trcldf_OFF        
# endif
   !
   USE in_out_manager ! I/O manager
   USE prtctl         ! Print control
   USE lbclnk         ! ocean lateral boundary conditions (or mpp link)
   USE lib_mpp        ! MPP library
   USE timing         ! Timing
   USE iom            ! I/O manager library

   IMPLICIT NONE
   PRIVATE

   PUBLIC   tra_zdf       ! called by stprk3_stg.F90
   PUBLIC   tra_zdf_imp   ! called by trczdf.F90

   !                                    !!!*  output flags for the zdf implicit trend diagnostics 
   LOGICAL ::   lo_zdf_tot   = .FALSE.   ! total vertical trend
   LOGICAL ::   lo_zdf_avt   = .FALSE.   ! vertical diffusion (avt=cst,issue from TKE, GLS, tides)
   LOGICAL ::   lo_zdf_evd   = .FALSE.   ! vertical neutral diffusion
   LOGICAL ::   lo_zdf_ldf   = .FALSE.   ! vertical neutral diffusion
   LOGICAL ::   lo_zdf_zad   = .FALSE.   ! vertical advection
   LOGICAL ::   lo_zdf_mfc   = .FALSE.   ! mass flux convection 
   LOGICAL ::   lo_zdf_split = .FALSE.   ! at least 1 split e3t*trend

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE tra_zdf( kt, Kbb, Kmm, Krhs, pts, Kaa )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE tra_zdf  ***
      !!
      !! ** Purpose :   compute the vertical ocean tracer physics.
      !!
      !! ** Action  : - ts(Naa)  T and S fields at time-step N+1 
      !!              - send trends to trdtra module for further diagnostics(l_trdtra=T)
      !!---------------------------------------------------------------------
      INTEGER                                  , INTENT(in)    :: kt                  ! ocean time-step index
      INTEGER                                  , INTENT(in)    :: Kbb, Kmm, Krhs, Kaa ! time level indices
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt), INTENT(inout) :: pts                 ! active tracers and RHS of tracer equation
      !
      INTEGER  ::   ji, jj, jk, jn   ! Dummy loop indices
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   ztrdts   ! 4D work space 
      !!---------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('tra_zdf')
      !
      IF( kt == nit000 ) THEN
         IF( .NOT. l_istiled .OR. ntile == 1 ) THEN   ! Do only on the first tile
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) 'tra_zdf : implicit vertical mixing on T & S'
            IF(lwp) WRITE(numout,*) '~~~~~~~'
         ENDIF
         !
!!gm     ! As far as I understand, trends diagnostics are currently incompatible with tiling    
         !
!!gm to be done : ===>>> TRC case define lo_zdf_xxx  logical in trczdf.F90 module   (To be done with Christian Ethe)
         !
         IF( l_trdtra ) THEN   ! thickness weighted tracer trend diagnostics 
            lo_zdf_tot   =                     iom_use("ttrd_zdf_tot") .OR. iom_use("strd_zdf_tot")   ! total vertical trend
            lo_zdf_avt   =                     iom_use("ttrd_zdf_avt") .OR. iom_use("strd_zdf_avt")   ! vertical diffusion (including evd)
            lo_zdf_evd   =                     iom_use("ttrd_zdf_evd") .OR. iom_use("strd_zdf_evd")   ! enhance vertical diffusion
            lo_zdf_ldf   = l_ldfslp    .AND. ( iom_use("ttrd_zdf_ldf") .OR. iom_use("strd_zdf_ldf") ) ! vertical neutral diffusion
            lo_zdf_zad   = ln_zad_Aimp .AND. ( iom_use("ttrd_zdf_zad") .OR. iom_use("strd_zdf_zad") ) ! vertical advection
            lo_zdf_mfc   = ln_zdfmfc   .AND. ( iom_use("ttrd_zdf_mfc") .OR. iom_use("strd_zdf_mfc") ) ! mass flux convection 
            lo_zdf_split = lo_zdf_avt .OR. lo_zdf_evd .OR. lo_zdf_ldf .OR. lo_zdf_zad .OR. lo_zdf_mfc ! at least 1 split trend
         ENDIF
      ENDIF
      !
      IF( lo_zdf_tot ) THEN    ! total implicit e3t*trends: store input T-S e3t*trends (i.e. RHS)
         ALLOCATE( ztrdts(A2D(0),jpk,jpts) )
         DO jn = 1,jpts
            DO_3D( 0, 0, 0, 0, 1, jpk )
               ztrdts(ji,jj,jk,jn) = pts(ji,jj,jk,jn,Krhs)
            END_3D
         END DO   
      ENDIF
      !
      !           !==  compute the implicit vertical trend  ==!      ! output : pts(Kaa) 
      !
      CALL tra_zdf_imp( 'TRA', rDt, Kbb, Kmm, Krhs, pts, Kaa, jpts )

!!smg================================================================
!!gm WHY here !   and I don't like that !
      !                             !**  DRAKKAR SSS control  **!
      !                                   ! JMM avoid negative salinities near river outlet ! Ugly fix
      !                                   ! JMM : restore negative salinities to small salinities:
      !                                   !!jc: discard this correction in case salinity is not used in eos
      IF( .NOT.( ln_SEOS .AND. rn_b0==0._wp ) ) THEN
         WHERE( pts(T2D(0),:,jp_sal,Kaa) < 0._wp )   pts(T2D(0),:,jp_sal,Kaa) = 0.1_wp
      ENDIF
!!smg================================================================
!!smg: we should instead put a threshold on salinity in the EOS.
!!smg: this patch breaks salt conservation.      
!!gm   in addition, it stpctl check global min(SSS) at add a warning 
!!gm   or better provide the number of time step and max of the SSS negative values 
!!smg================================================================      

      !           !==  save all T-S components of time-implicit vertical e3t*trends  ==!
      !
      IF( lo_zdf_tot ) THEN         !**  total vertical e3t*trends  **!   ( dt[T] - input trend )
         DO jn = 1, jpts
            DO_3D( 0, 0, 0, 0, 1, jpkm1 )    !-  dt[T] - input trend  -!
               ztrdts(ji,jj,jk,jn) = (  pts(ji,jj,jk,jn,Kaa) * e3t(ji,jj,jk,Kaa)    &
                  &                   - pts(ji,jj,jk,jn,Kbb) * e3t(ji,jj,jk,Kbb)  ) * r1_Dt   &
                  &                - ztrdts(ji,jj,jk,jn)
            END_3D
            CALL trd_tra( kt, Kmm, Krhs, 'TRA', jn, jptra_zdf_tot, ztrdts(:,:,:,jn) )
         END DO
         DEALLOCATE( ztrdts )
      ENDIF
      !
      !                             !**  split e3t*trends  **!
      !
      IF( lo_zdf_split )   CALL tra_zdf_trd( kt, Kmm, Kaa, 'TRA', pts )
      !
      IF( ln_timing )   CALL timing_stop('tra_zdf')
      !
   END SUBROUTINE tra_zdf


   SUBROUTINE tra_zdf_trd( kt, Kmm, Kaa, ctype, pts )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE tra_zdf_trd  ***
      !!
      !! ** Purpose :   diagnose split vertical ocean tracer physics e3t*trends.
      !!
      !! ** Action  : - ts(Naa)  T and S fields at time-step N+1 
      !!              - send trends to trdtra module for further diagnostics(l_trdtra=T)
      !!---------------------------------------------------------------------
      INTEGER                                  , INTENT(in)    ::   kt, Kmm, Kaa   ! time-step & -level indices
      CHARACTER(len=3)                         , INTENT(in   ) ::   ctype          ! =TRA or TRC (tracer indicator)
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt), INTENT(inout) ::   pts            ! active tracers and RHS of tracer equation
      !
      INTEGER  ::   ji, jj, jk, jn   ! Dummy loop indices
      REAL(wp) ::   zFw_kp1          ! local scalar
      REAL(wp), DIMENSION(A2D(0))               ::   zFw      ! 2D workspace
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   ztrdts   ! 4D workspace
!!gm      REAL(wp), DIMENSION(:,:,:)  , POINTER     ::   zav      ! 3D pointer => avt or avs
      !!---------------------------------------------------------------------
!     !!------------------!!-----------------------!!
!     !!     jp-names     !!     xml nicknames     !!
!     !!------------------!!-----------------------!!
!     !!  jptra_zdf_tot   !!   ttrd/strd_zdf_tot   !!
!     !!  jptra_zdf_avt   !!   ttrd/strd_zdf_avt   !!
!     !!  jptra_zdf_evd   !!   ttrd/strd_zdf_evd   !!
!     !!  jptra_zdf_ldf   !!   ttrd/strd_zdf_ldf   !!
!     !!  jptra_zdf_zad   !!   ttrd/strd_zdf_zad   !!
!     !!  jptra_zdf_mfc   !!   ttrd/strd_zdf_mfc   !!   
!     !!------------------!!-----------------------!!
      
!!smg2gm     manage 'TRC'  e3t*trends

      IF( lo_zdf_split )  THEN
         ALLOCATE( ztrdts(A2D(0),jpk,jpts) )
         ztrdts(:,:,:,:) = 0._wp
      ENDIF
      
      IF( lo_zdf_avt ) THEN            !-  vertical diffusion (including evd)  -!
         CALL zdf_trd_imp( kt, Kmm, Kaa, 'TRA', 1, jptra_zdf_avt, pts(A2D(0),:,1,Kaa), avt(A2D(0),:) ) 
         CALL zdf_trd_imp( kt, Kmm, Kaa, 'TRA', 2, jptra_zdf_avt, pts(A2D(0),:,2,Kaa), avs(A2D(0),:) ) 
      ENDIF
      !
      IF( lo_zdf_evd ) THEN            !-  enhance vertical diffusion  -!
         DO jn = 1, jpts
            CALL zdf_trd_imp( kt, Kmm, Kaa, 'TRA', jn, jptra_zdf_evd, pts(A2D(0),:,jn,Kaa), avt_evd(A2D(0),:) ) 
         END DO
      ENDIF
      !
      IF( lo_zdf_ldf ) THEN            !-   vertical neutral diffusion  -!
         DO jn = 1, jpts
            IF( ln_traldf_msc ) THEN         ! Method of Stabilizing Correction
               CALL zdf_trd_imp( kt, Kmm, Kaa, 'TRA', jn, jptra_zdf_ldf, pts(A2D(0),:,jn,Kaa), akz(A2D(0),:) ) 
            ELSE                             ! Full implicit 
               CALL zdf_trd_imp( kt, Kmm, Kaa, 'TRA', jn, jptra_zdf_ldf, pts(A2D(0),:,jn,Kaa), ah_wslp2(A2D(0),:) )
            ENDIF 
         END DO
      ENDIF
         !
      IF( lo_zdf_zad ) THEN      !**  Adaptive implicit vertical advection  **!
         !
         DO jn = 1, jpts                  !- loop over the T & S
               !
!!gm2/smg :    deal with linear free surf case ? : manage or remove it from trend diag ??
!!gm           linear case zFw(:,:) = wi(:,:) * pt(:,:,1)
!!gm           caution in nonlinear ssh, the surface vertical velocity (ww) is not necessary 0 
!!gm           as ww is computed with the sum of divh from the bottom : should we compute this error ?
!!gm           !
            !                                ! linear free-surf. non-zero flux otherwise 0 
            IF( lk_linssh ) THEN   ;   zFw(:,:) = wi(T2D(0),1) * pts(T2D(0),1,jn,Kmm)
            ELSE                   ;   zFw(:,:) = 0._wp
            ENDIF
            !
            DO jk = 1, jpk-2
               DO_2D( 0, 0, 0, 0 )
                  !                             ! implicit vertical advective fluxes at level jk+1
                  zFw_kp1 = MAX( wi(ji,jj,jk+1) , 0._wp ) * pts(ji,jj,jk  ,jn,Kaa)   &   ! no mask as wi is w-masked
                     &    + MIN( wi(ji,jj,jk+1) , 0._wp ) * pts(ji,jj,jk+1,jn,Kaa)
                  !                             ! thickness weighted trend
                  ztrdts(ji,jj,jk,jn) = ( zFw(ji,jj) - zFw_kp1 )
                  !                             ! save vertical flux for jk+1 computation
                  zFw(ji,jj)         = zFw_kp1
               END_2D
            END DO
            ztrdts(:,:,jpkm1,jn) = zFw(:,:)         ! no flux at level jpk
            !
            CALL trd_tra( kt, Kmm, Kaa, 'TRA', jn, jptra_zdf_zad, ztrdts(:,:,:,jn) )
            !
         END DO                !- loop over the T & S
         !
      ENDIF
      !
      IF( lo_zdf_mfc ) THEN            !  
         !
         DO jn = 1, jpts                  !- loop over the T & S
            !
!!gm     CAUTION rDt should be pass in argument (/= for passive tracers)
            DO jk = 1, jpkm1
               DO_2D( 0, 0, 0, 0 )
                  ztrdts(ji,jj,jk,jn) = edmftra(ji,jj,jk,jn)  * r1_Dt                        & ! plume explicit contribution directly added to the RHS 
                     &                - (   edmfm(ji,jj,jk  ) * pts(ji,jj,jk  ,jn,Kaa)       & ! remaining water implicit contribution
                     &                    - edmfm(ji,jj,jk+1) * pts(ji,jj,jk+1,jn,Kaa)   )   &
                     &                / e3w(ji,jj,jk+1,Kmm) 
               END_2D
            END DO
            !
!!gm the why of the contribution 
            ! zwd(ji,jk) = - e3t(ji,jj,jk,Kaa) * pdt * edmfm(ji,jj,jk  ) / e3w(ji,jj,jk+1,Kmm)
            ! zws(ji,jk) = + e3t(ji,jj,jk,Kaa) * pdt * edmfm(ji,jj,jk+1) / e3w(ji,jj,jk+1,Kmm)
            !
            ! matrix :   zwd(jk) * t(jk,Kaa) - zws(j) * t(jk+1,Kaa)
            !
            !         = - e3t(ji,jj,jk,Kaa) * pdt * edmfm(ji,jj,jk  ) / e3w(ji,jj,jk+1,Kmm) * t(jk  ,Kaa)
            !           + e3t(ji,jj,jk,Kaa) * pdt * edmfm(ji,jj,jk+1) / e3w(ji,jj,jk+1,Kmm) * t(jk+1,Kaa)
            !
            !         = - e3t(ji,jj,jk,Kaa) / e3w(ji,jj,jk+1,Kmm) * pdt 
            !           * (  edmfm(ji,jj,jk  ) * t(jk  ,Kaa)
            !              - edmfm(ji,jj,jk+1) * t(jk+1,Kaa) )
            ! 
            ! e3t*trend is thus : - (  edmfm(ji,jj,jk  ) * t(jk  ,Kaa)
            !                        - edmfm(ji,jj,jk+1) * t(jk+1,Kaa) ) / e3w(ji,jj,jk+1,Kmm) 
         END DO
      ENDIF
      !
      DEALLOCATE( ztrdts )   
      !
   END SUBROUTINE tra_zdf_trd


   SUBROUTINE zdf_trd_imp( kt, Kmm, Kaa, ctype, ktra, ktrd, pta, pKz )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE zdf_trd_imp  ***
      !!
      !! ** Purpose :   diagnose of vertical implicit diffusive e3t*trend 
      !!              from a tracer field at Kaa (after time-step) and
      !!              send it to trd_tra.
      !!
      !! ** Method  : - The implicit vertical diffusion e3t*trend of a tracer T
      !!             is given by:
      !!               e3t * dz( avt dz(T) ) <=> dk[ avt/e3w dk[T] ]
      !!             where T is taken at Kaa (after time step) 
      !!
      !! ** Action  : - provide trends to trdtra module (l_trdtra=T)
      !!---------------------------------------------------------------------
      INTEGER                   , INTENT(in   ) ::   kt, Kmm, Kaa   ! time -step & -level indices
      CHARACTER(len=3)          , INTENT(in   ) ::   ctype          ! tracers trends type 'TRA'/'TRC'
      INTEGER                   , INTENT(in   ) ::   ktra, ktrd     ! tracer and trend indices
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in   ) ::   pta       ! cdtype tracer at N+1 (after)   [cdtype units]
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in   ) ::   pKz       ! vertical diffusive coefficient [m2/s]
      !!
      INTEGER  ::   ji, jj, jk      ! local integers 
      REAL(wp) ::   zFw_kp1         ! local scalar
      REAL(wp), DIMENSION(A2D(0)    ) ::   zFw    ! 2D diffusive Flux at w-points (level jk)
      REAL(wp), DIMENSION(A2D(0),jpk) ::   ztrd   ! implicit diffusive trend 
      !!---------------------------------------------------------------------
      ! DIMENSION(A2D(nn_hls),jpk)
      ! optional arguments?  pta0,pta_hls,pkz,pkz_hls 
      
      zFw(:,:) = 0._wp                    ! no diffusive flux through the surface
      !
      DO jk = 1, jpk-2                    ! pKz has been masked by wmask
         !
         DO_2D( 0, 0, 0, 0 )
            zFw_kp1 = pKz(ji,jj,jk+1) * (   pta(ji,jj,jk  )   &                     ! k+1 flux
               &                          - pta(ji,jj,jk+1)   ) / e3w(ji,jj,jk+1, Kmm)
            ztrd(ji,jj,jk) = (  zFw(ji,jj) - zFw_kp1  )                                      ! level k e3t*trend
            zFw(ji,jj)     = zFw_kp1                                                         ! store next level calculus
         END_2D
         !
      END DO
      ztrd(:,:,jpkm1) = zFw(:,:)   ! zFw(jpk) = jpk
      ztrd(:,:,jpk  ) = 0._wp      ! T-level at jpk is always land                                           
      !
      CALL trd_tra( kt, Kmm, Kaa, ctype, ktra, ktrd, ztrd )
      !
   END SUBROUTINE zdf_trd_imp

!!gm
!!gm =====>   end of    tra_zdf_trd  and  zdf_trd_imp  TO BE MOVED after tra_zdf_imp routine
!!gm

   SUBROUTINE tra_zdf_imp( cdtype, pdt, Kbb, Kmm, Krhs, pt, Kaa, kjpt )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE tra_zdf_imp  ***
      !!
      !! ** Purpose :   Compute the after tracer through a implicit computation
      !!     of the vertical tracer diffusion (including the vertical component
      !!     of lateral mixing (only for 2nd order operator, for fourth order
      !!     it is already computed and add to the general trend in traldf)
      !!
      !! ** Method  :  The vertical diffusion of a tracer ,t , is given by:
      !!          difft = dz( avt dz(t) ) = 1/e3t dk+1( avt/e3w dk(t) )
      !!      It is computed using a backward time scheme (t=after field)
      !!      which provide directly the after tracer field.
      !!      If ln_zdfddm=T, use avs for salinity or for passive tracers
      !!      Surface and bottom boundary conditions: no diffusive flux on
      !!      both tracers (bottom, applied through the masked field avt).
      !!      If iso-neutral mixing, add to avt the contribution due to lateral mixing.
      !!
      !! ** Action  : - pt(:,:,:,:,Kaa)  becomes the after tracer
      !!---------------------------------------------------------------------
      INTEGER                                  , INTENT(in   ) ::   Kbb, Kmm, Krhs, Kaa  ! ocean time level indices
      CHARACTER(len=3)                         , INTENT(in   ) ::   cdtype   ! =TRA or TRC (tracer indicator)
      INTEGER                                  , INTENT(in   ) ::   kjpt     ! number of tracers (jpts or jptra)
      REAL(wp)                                 , INTENT(in   ) ::   pdt      ! tracer time-step
      REAL(wp), DIMENSION(jpi,jpj,jpk,kjpt,jpt), INTENT(inout) ::   pt       ! tracers and its thickness weighted RHS
      !
      INTEGER  ::  ji, jj, jk, jn   ! dummy loop indices
      REAL(wp) ::  zrhs, zzwi, zzws ! local scalars
      REAL(wp), DIMENSION(T1Di(0),jpk) ::  zwi, zwt, zwd, zws
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   ztrdts   ! 4D workspace
!!gm      REAL(wp), DIMENSION(:,:,:)  , POINTER     ::   zav      ! 3D pointer => avt or avs idea to be tested...
      !!---------------------------------------------------------------------
      !
      IF( lo_zdf_tot ) THEN    ! total implicit e3t*trends: store input tracer e3t*trends (i.e. RHS)
         ALLOCATE( ztrdts(A2D(0),jpk,kjpt) )
          DO jn = 1,kjpt
            DO_3D( 0, 0, 0, 0, 1, jpk )
               ztrdts(ji,jj,jk,jn) = pt(ji,jj,jk,jn,Krhs)
            END_3D
         END DO 
      ENDIF
      
      !                                               ! ================= !
      DO_1Dj( 0, 0 )                                  !    i-k slices     !   ( jj-loop )
         !                                            ! ================= !
         DO jn = 1, kjpt                              !    tracer loop    !
            !                                         ! ================= !
            !
            !  Matrix construction
            ! --------------------
            ! Build matrix if temperature or salinity (only in double diffusion case) or first passive tracer
            !
            IF(  ( cdtype == 'TRA' .AND. ( jn == jp_tem .OR. ( jn == jp_sal .AND. ln_zdfddm ) ) ) .OR.   &
               & ( cdtype == 'TRC' .AND. jn == 1 )  )  THEN
               !
               ! vertical mixing coef.: avt for temperature, avs for salinity and passive tracers
               !
               IF( cdtype == 'TRA' .AND. jn == jp_tem ) THEN     ! use avt  for temperature
                  !
                  IF( l_ldfslp ) THEN            ! use avt + neutral diffusion contribution
                     IF( ln_traldf_msc  ) THEN        ! Method of Stabilizing Correction
                        DO_2Dik( 0, 0,   2, jpk, 1 )
                           zwt(ji,jk) = avt(ji,jj,jk) + akz(ji,jj,jk)
                        END_2D
                     ELSE                             ! standard or triad neutral operator
                        DO_2Dik( 0, 0,   2, jpk, 1 )
                           zwt(ji,jk) = avt(ji,jj,jk) + ah_wslp2(ji,jj,jk)
                        END_2D
                     ENDIF
                  ELSE                          ! use avt only
                     DO_2Dik( 0, 0,   2, jpk, 1 )
                        zwt(ji,jk) = avt(ji,jj,jk)
                     END_2D
                  ENDIF
                  !
               ELSE                                               ! use avs for salinity or passive tracers
                  !
# if defined key_top
                  IF( l_ldfslp.AND.(.NOT.ln_trcldf_OFF) ) THEN    ! use avs + neutral diffusion contribution
# else
                  IF( l_ldfslp ) THEN                             ! use avs + neutral diffusion contribution
# endif
                     IF( ln_traldf_msc  ) THEN        ! MSC neutral operator
                        DO_2Dik( 0, 0,   2, jpk, 1 )
                           zwt(ji,jk) = avs(ji,jj,jk) + akz(ji,jj,jk)
                        END_2D
                     ELSE                             ! standard or triad neutral operator
                        DO_2Dik( 0, 0,   2, jpk, 1 )
                           zwt(ji,jk) = avs(ji,jj,jk) + ah_wslp2(ji,jj,jk)
                        END_2D
                     ENDIF
                  ELSE                          !
                     DO_2Dik( 0, 0,   2, jpk, 1 )
                        zwt(ji,jk) = avs(ji,jj,jk)
                     END_2D
                  ENDIF
               ENDIF
               zwt(:,1) = 0._wp
               !
               ! Diagonal, lower (i), upper (s)  (including the bottom boundary condition since avt is masked)
               IF( ln_zad_Aimp ) THEN         ! Adaptive implicit vertical advection
                  DO_2Dik( 0, 0,   1, jpkm1, 1 )
                     zzwi = - pdt * zwt(ji,jk  ) / e3w(ji,jj,jk  ,Kmm)
                     zzws = - pdt * zwt(ji,jk+1) / e3w(ji,jj,jk+1,Kmm)
                     zwd(ji,jk) = e3t(ji,jj,jk,Kaa) - ( zzwi + zzws )   &
                        &              + pdt * ( MAX( wi(ji,jj,jk  ) , 0._wp ) &
                        &                      - MIN( wi(ji,jj,jk+1) , 0._wp ) )
                     zwi(ji,jk) = zzwi + pdt *   MIN( wi(ji,jj,jk  ) , 0._wp )
                     zws(ji,jk) = zzws - pdt *   MAX( wi(ji,jj,jk+1) , 0._wp )
                  END_2D
               ELSE
                  DO_2Dik( 0, 0,   1, jpkm1, 1 )
                     zwi(ji,jk) = - pdt * zwt(ji,jk  ) / e3w(ji,jj,jk,Kmm)
                     zws(ji,jk) = - pdt * zwt(ji,jk+1) / e3w(ji,jj,jk+1,Kmm)
                     zwd(ji,jk) = e3t(ji,jj,jk,Kaa) - ( zwi(ji,jk) + zws(ji,jk) )
                  END_2D
               ENDIF
               !
!!gm  BUG?? : if edmfm is equivalent to a w  ==>>>   just add +/-  rDt * edmfm(ji,jj,jk+1/jk  )
!!            but edmfm is at t-point !!!!   crazy???  why not keep it at w-point????
!!gm   BUG ???   below  e3t_Kmm  should be used ?
!!               or even no multiplication by e3t unless there is a bug in wi calculation
               IF( ln_zdfmfc ) THEN    ! add upward Mass Flux in the matrix
                  DO_2Dik( 0, 0,   1, jpkm1, 1 )
                     ! zwi not updated- in the original zdfmfc.F90 calculation the added flux was zero over 1:jpkm1
                     zws(ji,jk) = zws(ji,jk) + e3t(ji,jj,jk,Kaa) * pdt * edmfm(ji,jj,jk+1) / e3w(ji,jj,jk+1,Kmm)
                     zwd(ji,jk) = zwd(ji,jk) - e3t(ji,jj,jk,Kaa) * pdt * edmfm(ji,jj,jk  ) / e3w(ji,jj,jk+1,Kmm)
                  END_2D
               ENDIF
               !
               !! Matrix inversion from the first level
               !!----------------------------------------------------------------------
               !   solve m.x = y  where m is a tri diagonal matrix ( jpk*jpk )
               !
               !        ( zwd1 zws1   0    0    0  )( zwx1 ) ( zwy1 )
               !        ( zwi2 zwd2 zws2   0    0  )( zwx2 ) ( zwy2 )
               !        (  0   zwi3 zwd3 zws3   0  )( zwx3 )=( zwy3 )
               !        (        ...               )( ...  ) ( ...  )
               !        (  0    0    0   zwik zwdk )( zwxk ) ( zwyk )
               !
               !   m is decomposed in the product of an upper and lower triangular matrix.
               !   The 3 diagonal terms are in 3d arrays: zwd, zws, zwi.
               !   Suffices i,s and d indicate "inferior" (below diagonal), diagonal
               !   and "superior" (above diagonal) components of the tridiagonal system.
               !   The solution will be in the 4d array pta.
               !   The 3d array zwt is used as a work space array.
               !   En route to the solution pt(:,:,:,:,Kaa) is used a to evaluate the rhs and then
               !   used as a work space array: its value is modified.
               !
               DO_1Di( 0, 0 )          !* 1st recurrence:   Tk = Dk - Ik Sk-1 / Tk-1   (increasing k) ! done one for all passive tracers (so included in the IF instruction)
                  zwt(ji,1) = zwd(ji,1)
               END_1D
               DO_2Dik( 0, 0,   2, jpkm1, 1 )
                  zwt(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwt(ji,jk-1)
               END_2D
               !
            ENDIF
            !
            IF( ln_zdfmfc ) THEN    ! add Mass Flux to the RHS
               DO_2Dik( 0, 0,   1, jpkm1, 1 )
                  pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) + edmftra(ji,jj,jk,jn)   !!smg to be fixed by e3t multiplier in mfc calculation
               END_2D
            ENDIF
            !
            DO_1Di( 0, 0 )             !* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1
            pt(ji,jj,1,jn,Kaa) =       e3t(ji,jj,1,Kbb) * pt(ji,jj,1,jn,Kbb )    &
               &               + pdt                    * pt(ji,jj,1,jn,Krhs)
            END_1D
            DO_2Dik( 0, 0,   2, jpkm1, 1 )
               zrhs =       e3t(ji,jj,jk,Kbb) * pt(ji,jj,jk,jn,Kbb )   &
                  & + pdt                     * pt(ji,jj,jk,jn,Krhs)   ! zrhs=thickness weighted right hand side 
               pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)
            END_2D
            !
            DO_1Di( 0, 0 )             !* 3d recurrence:    Xk = (Zk - Sk Xk+1 ) / Tk   (result is the after tracer)
               pt(ji,jj,jpkm1,jn,Kaa) = pt(ji,jj,jpkm1,jn,Kaa) / zwt(ji,jpkm1) * tmask(ji,jj,jpkm1)
            END_1D
            DO_2Dik( 0, 0,   jpk-2, 1, -1 )
               pt(ji,jj,jk,jn,Kaa) = ( pt(ji,jj,jk,jn,Kaa) - zws(ji,jk) * pt(ji,jj,jk+1,jn,Kaa) )   &
                  &             / zwt(ji,jk) * tmask(ji,jj,jk)
            END_2D
            !                                         ! ================= !
         END DO                                       !    tracer loop    !
         !                                            ! ================= !
      END_1D                                          !    i-k slices     !   ( jj-loop )
      !                                               ! ================= !
   END SUBROUTINE tra_zdf_imp

   !!==============================================================================
END MODULE trazdf
