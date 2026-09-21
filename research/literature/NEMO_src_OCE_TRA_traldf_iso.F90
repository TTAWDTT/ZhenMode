MODULE traldf_iso
   !!======================================================================
   !!                   ***  MODULE  traldf_iso  ***
   !! Ocean  tracers:  horizontal component of the lateral tracer mixing trend
   !!======================================================================
   !! History :  OPA  ! 1994-08  (G. Madec, M. Imbard)
   !!            8.0  ! 1997-05  (G. Madec)  split into traldf and trazdf
   !!            NEMO ! 2002-08  (G. Madec)  Free form, F90
   !!            1.0  ! 2005-11  (G. Madec)  merge traldf and trazdf :-)
   !!            3.3  ! 2010-09  (C. Ethe, G. Madec) Merge TRA-TRC
   !!            3.7  ! 2014-01  (G. Madec, S. Masson)  restructuration/simplification of aht/aeiv specification
   !!             -   ! 2014-02  (F. Lemarie, G. Madec)  Standard and triad operator with Method of Stabilizing Correction
   !!            4.5  ! 2022-06  (S. Techene, G, Madec)  refactorization to reduce local memory usage + add tra_ldf_iso_a33 &
   !!                 !                                  traldf_iso_blp routines and traldf_iso_scheme.h90 file
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   tra_ldf_iso   : update the tracer trend with the horizontal component of a iso-neutral laplacian operator
   !!                   and with the vertical part of the isopycnal or geopotential s-coord. operator
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and active tracers
   USE dom_oce        ! ocean space and time domain
   USE domutl, ONLY : lbnd_ij
   USE trc_oce        ! share passive tracers/Ocean variables
   USE zdf_oce        ! ocean vertical physics
   USE ldftra         ! lateral diffusion: tracer eddy coefficients
   USE ldfslp         ! iso-neutral slopes
   USE diaptr         ! poleward transport diagnostics
   USE diaar5         ! AR5 diagnostics
   !
   USE in_out_manager ! I/O manager
   USE iom            ! I/O library
   USE phycst         ! physical constants
   USE lbclnk         ! ocean lateral boundary conditions (or mpp link)
   USE lib_fortran    ! to use sign with key_nosignedzero

   IMPLICIT NONE
   PRIVATE

   PUBLIC   traldf_iso_lap   ! routine called by traadv.F90
   PUBLIC   traldf_iso_blp   ! routine called by traadv.F90
   PUBLIC   traldf_iso_a33   ! routine called by traldf_iso.F90    !!gm: to be move in traadv.F90 ???
!!gm                                                               !!gm: to be extended to 13 and 23 ???)

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE traldf_iso_lap( kt, Kbb, Kmm, pt, Krhs, ld_ptr, ld_hst )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE traldf_iso_lap  ***
      !!
      !!                   ——    nn_hls =2 or more  ——
      !!
      !! ** Purpose :   Compute the before horizontal tracer (t & s) diffusive
      !!      trend for a laplacian tensor (ezxcept the dz[ dz[.] ] term) and
      !!      add it to the general trend of tracer equation.
      !!
      !! ** Method  :   The horizontal component of the lateral diffusive trends
      !!      is provided by a 2nd order operator rotated along neural or geopo-
      !!      tential surfaces to which an eddy induced advection can be added
      !!      It is computed using before fields (forward in time) and isopyc-
      !!      nal or geopotential slopes computed in routine ldfslp.
      !!
      !!                        (  A11   0   A13  )
      !!      rotation matrix = (   0   A22  A23  )
      !!                        (  A31  A32  A33  )
      !!
      !!      • masked horizontal derivative of T  ( di[ t ] )
      !!
      !!      •  horizontal fluxes of the lateral mixing operator
      !!         zfu =  A11 di[ tb ] + A13  dk[ mi(mk(tb)) ]
      !!         zfv =  ahtv e1v*e3v/e2v dj[ tb ]
      !!              - ahtv e2u*vslp    dk[ mj(mk(tb)) ]
      !!        with  A11 = ahtu e2u*e3u/e1u  ;  A13 = - ahtu e2u*uslp
      !!              A22 = ahtv e1v*e3v/e1v  ;  A23 = - ahtv e1v*vslp
      !!      take the horizontal divergence of the fluxes:
      !!         difft = 1/(e1e2t*e3t) {  di-1[ zfu ] +  dj-1[ zfv ]  }
      !!      Add this trend to the general trend (ta,sa):
      !!         ta = ta + difft
      !!
      !!      • vertical trends of the lateral mixing operator
      !!       (excluding the implicit part of vertical flux proportional to dk[t] )
      !!      vertical fluxes associated with the rotated lateral mixing:
      !!         zfw = - {  mi(mk(ahtu)) * e2t*wslpi di[ mi(mk(tb)) ]
      !!                  + mj(mk(ahtv)) * e1t*wslpj dj[ mj(mk(tb)) ]  }
      !!      take the horizontal divergence of the fluxes:
      !!         difft = 1/(e1e2t*e3t) dk[ zfw ]
      !!      Add this trend to the general trend (ta,sa):
      !!         pt_rhs = pt_rhs + difft
      !!
      !! ** Action :   Update pt_rhs arrays with the before rotated diffusion
      !!----------------------------------------------------------------------
      INTEGER                       , INTENT(in   ) ::   kt, Kbb, Kmm, Krhs   ! ocean time-step and time-level indices
      LOGICAL , OPTIONAL            , INTENT(in   ) ::   ld_hst, ld_ptr       ! T-S diagnostic flags
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(inout) ::   pt                   ! tracers, in: at kbb ; out: at Krhs
      !!
      INTEGER  ::   ji, jj, jk, jn     ! dummy loop indices
      INTEGER  ::   inn                ! inner domain index
      INTEGER  ::   itra               ! number of tracers
      INTEGER  ::   ik, ikp1, iis      ! swap  indices
      !
      LOGICAL  ::   ll_ptr, ll_hst
      !
      REAL(wp) ::   zmsku, zahu_w      ! local scalars
      REAL(wp) ::   zmskv, zahv_w      !   -      -
      REAL(wp) ::   zfw_kp1            !   -      -
      REAL(wp) ::   zA11      , zA13   !)
      REAL(wp) ::         zA22, zA23   !) elements of the rotation matrix
      REAL(wp) ::   zA31, zA32, zA33   !)
      !
      REAL(wp), DIMENSION(T2D(1),0:1) ::   zdit, zdjt   ! INNER + 1 domain at level jk and jk+1
      REAL(wp), DIMENSION(T2D(1),0:1) ::   zdkt         ! INNER + 1 domain at level jk and jk+1
      REAL(wp), DIMENSION(T2D(1)    ) ::   zfu , zfv    ! INNER + 1 domain
      REAL(wp), DIMENSION(T2D(0)    ) ::   zfw          ! INNER     domain
      !!----------------------------------------------------------------------
      ll_ptr = .FALSE. ; ll_hst = .FALSE.
      IF( PRESENT(ld_ptr) ) ll_ptr = l_diaptr .AND. ld_ptr
      IF( PRESENT(ld_hst) ) ll_hst = ld_hst
      !
!!gm OPTIMIZATION : This part does not depends on tracer  ===>>> put in a routine
!!                  possibility of moving it in tra_ldf routine (shared between TRA and TRC at least in RK3 case).
!!gm idea:  define A13 and A23 as 2D arrays that do not depends on tracers in tra_ldf_iso_a33 routine which
!!gm        will be renamed : tra_ldf_iso_a13_23_33    ===>> 2x3D additional arrays but much less calculation...
!!gm idea 2 : even better: explore the possibility to compute matrix element instead of slopes in ldfslp ....
!!gm                                                   ===>> same memory print but much less calculation !!
!!            caution: the gm velocity have also to be calculated as they use the slopes....
!!gm idea
      !
      CALL traldf_iso_a33( Kmm, ah_wslp2, akz )   ! calculate  a33 element   (ah_wslp2 and akz)
      !
      itra = SIZE( pt, dim = 4 )                   ! number of tracers
      !
      DO jn = 1, itra                     !==  tracer loop  ==!
         !
         DO jk = 1, jpkm1                       !=  ij slab  =!
            !
            !                                      !* iso-neutral laplacian applied on pt over the INNER domain
#           define   iso_lap
#           define   INN                0
#           define   pt_in(i,j,k,n,t)   pt(i,j,k,n,t)

#           include   "traldf_iso_scheme.h90"

#           undef    iso_lap
#           undef    INN
#           undef    pt_in
            !
            !
            !                                   !=  "Poleward" & 2D-integrated diffusive heat and salt transports  =!
            !                                       Note sign is reversed to give down-gradient diffusive transports
            IF( ll_ptr )  CALL dia_ptr_hst( jn, 'ldf', -zfv(:,:) )
            IF( ll_hst )  CALL dia_ar5_hst( jn, 'ldf', -zfu(:,:), -zfv(:,:), ldfin=(jk == jpkm1) )
         END DO                             !=  end ij slab  =!
         !
      END DO                          !==  end tracer loop  ==!
      !
   END SUBROUTINE traldf_iso_lap


   SUBROUTINE traldf_iso_blp( kt, Kbb, Kmm, pt, Krhs, ld_ptr, ld_hst )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE traldf_iso_blp  ***
      !!
      !!                   ——    nn_hls =2 or more  ——
      !!
      !!----------------------------------------------------------------------
      INTEGER                       , INTENT(in   ) ::   kt, Kbb, Kmm, Krhs   ! ocean time-step and time-level indices
      LOGICAL , OPTIONAL            , INTENT(in   ) ::   ld_hst, ld_ptr       ! T-S diagnostic flags
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(inout) ::   pt                   ! tracers, in: at kbb ; out: at Krhs
      !!
      INTEGER  ::   ji, jj, jk, jn     ! dummy loop indices
      INTEGER  ::   inn                ! inner domain index
      INTEGER  ::   itra               ! number of tracers
      INTEGER  ::   ik, ikp1, iis      ! swap  indices
      !
      LOGICAL  ::   ll_ptr, ll_hst
      !
      REAL(wp) ::   zmsku, zahu_w      ! local scalars
      REAL(wp) ::   zmskv, zahv_w      !   -      -
      REAL(wp) ::   zfw_kp1            !   -      -
      REAL(wp) ::   zA11      , zA13   !)
      REAL(wp) ::         zA22, zA23   !) elements of the rotation matrix
      REAL(wp) ::   zA31, zA32, zA33   !)
      !
      REAL(wp), DIMENSION(T2D(2),0:1)   ::   zdit, zdjt   ! INNER + 2 domain at level jk and jk+1
      REAL(wp), DIMENSION(T2D(2),0:1)   ::   zdkt         ! INNER + 2 domain at level jk and jk+1
      REAL(wp), DIMENSION(T2D(2)    )   ::   zfu , zfv    ! INNER + 2 domain
      REAL(wp), DIMENSION(T2D(1)    )   ::   zfw          ! INNER + 1 domain
      REAL(wp), DIMENSION(T2D(1),jpkm1) ::   zlap         ! INNER + 1 doamin (3D laplacian at t-point)
      !!----------------------------------------------------------------------
      ll_ptr = .FALSE. ; ll_hst = .FALSE.
      IF( PRESENT(ld_ptr) ) ll_ptr = l_diaptr .AND. ld_ptr
      IF( PRESENT(ld_hst) ) ll_hst = ld_hst
      !
      CALL traldf_iso_a33( Kmm, ah_wslp2, akz )   ! calculate  a33 element   (ah_wslp2 and akz)
      !
      itra = SIZE( pt, dim = 4 )                   ! number of tracers
      !
      DO jn = 1, itra                     !==  tracer loop  ==!
         !
         DO jk = 1, jpkm1                       !=  ij slab  =!
            !
            !                                      !* 1st pass : iso-neutral laplacian of pt
            !                                                    computed over the INNER + 1 domain
#           define   iso_blp_p1
#           define   INN                1
#           define   pt_in(i,j,k,n,t)   pt(i,j,k,n,t)
!
#           include "traldf_iso_scheme.h90"
!
#           undef    iso_blp_p1
#           undef    INN
#           undef    pt_in
            !
         END DO                             !=  end ij slab  =!
         !
         !
         DO jk = 1, jpkm1                       !=  ij slab  =!
            !
            !                                      !* 2nd pass : bilaplacian = laplacian of 1st pass (zlap)
            !                                                    computed over the INNER domain
#           define   iso_blp_p2
#           define   INN                0
#           define   pt_in(i,j,k,n,t)   zlap(i,j,k)
!
#           include "traldf_iso_scheme.h90"
!
#           undef    iso_blp_p2
#           undef    INN
#           undef    pt_in
            !
            !                                   !=  "Poleward" & 2D-integrated diffusive heat and salt transports  =!
            !                                       Note sign is reversed to give down-gradient diffusive transports
            IF( ll_ptr )  CALL dia_ptr_hst( jn, 'ldf', -zfv(:,:) )
            IF( ll_hst )  CALL dia_ar5_hst( jn, 'ldf', -zfu(:,:), -zfv(:,:), ldfin=(jk == jpkm1) )
         END DO                             !=  end ij slab  =!
         !
      END DO                          !==  end tracer loop  ==!
      !
   END SUBROUTINE traldf_iso_blp


   SUBROUTINE traldf_iso_a33( Kmm, pah_wslp2, pakz )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE traldf_iso_a33  ***
      !!
      !!                   ——    nn_hls =2 or more  ——
      !!
      !! ** Purpose :   Compute the a33 element of the rotation  matrix and,
      !!              if ln_traldf_msc=T, its split into explicit and implicit
      !!              time-integration.
      !!
      !! ** Method  :
      !!
      !! ** Action :   pah_wslp2, pakz
      !!----------------------------------------------------------------------
      INTEGER                   , INTENT(in   ) ::   Kmm         ! ocean time level index
      REAL(wp), DIMENSION(:,:,:), INTENT(inout) ::   pah_wslp2   ! implicit a33 element (ln_traldf_msc= false)
      REAL(wp), DIMENSION(:,:,:), INTENT(inout) ::   pakz        ! implicit a33 element (ln_traldf_msc= true )
      !
      INTEGER  ::  ji, jj, jk      ! dummy loop indices
      INTEGER  ::  inn             ! local integer
      REAL(wp) ::  zmsku, zahu_w   ! local scalars
      REAL(wp) ::  zmskv, zahv_w   !   -      -
      REAL(wp) ::  zcoef0, ze3w_2  !   -      -
      !!----------------------------------------------------------------------
      !
      SELECT CASE( nldf_tra )             ! set inner domain index
      CASE( np_lap_i  )   ;   inn  = 0
      CASE( np_blp_i  )   ;   inn  = 1
      CASE DEFAULT        ;   CALL ctl_stop( 'STOP', 'traldf_iso_a33 routine should not be called  ' )
      END SELECT
      !
      ! CAUTION:   round brackets are required for halo size and north fold compatibility
      !
      !
      DO_3D( inn , inn , inn , inn ,   2, jpkm1 )
         !
         zmsku = wmask(ji,jj,jk) / MAX(   ( umask(ji  ,jj,jk-1) + umask(ji-1,jj,jk) )          &
            &                           + ( umask(ji-1,jj,jk-1) + umask(ji  ,jj,jk) ) , 1._wp  )
         zmskv = wmask(ji,jj,jk) / MAX(   ( vmask(ji,jj  ,jk-1) + vmask(ji,jj-1,jk) )          &
            &                           + ( vmask(ji,jj-1,jk-1) + vmask(ji,jj  ,jk) ) , 1._wp  )
            !
         !                                   ! round brackets required to ensure halo size compatibility with north fold boundary condition
         zahu_w = (  ( ahtu(ji  ,jj,jk-1) + ahtu(ji-1,jj,jk) )    &
            &      + ( ahtu(ji-1,jj,jk-1) + ahtu(ji  ,jj,jk) )  ) * zmsku
         zahv_w = (  ( ahtv(ji,jj  ,jk-1) + ahtv(ji,jj-1,jk) )    &
            &      + ( ahtv(ji,jj-1,jk-1) + ahtv(ji,jj  ,jk) )  ) * zmskv
            !
         pah_wslp2(ji,jj,jk) = zahu_w * wslpi(ji,jj,jk) * wslpi(ji,jj,jk)   &
            &                + zahv_w * wslpj(ji,jj,jk) * wslpj(ji,jj,jk)
      END_3D
      !
!!gm size allocate for ah_wslp2 and akz should be A2D(0) in lap case and A2D(1) in bilap case


!!gm  NB:   ah_wslp2 and akz can be defined on A2D(0) or A2D(1) (lap, or bilap) and aver 2:jpkm1 in all cases
!!gm        moreover, without ln_traldf_msc=T  only ah_wslp2 is required  +  akz can be set as a local array
!!gm        if ah_wslp2 is set to a proper value at the end of tra_ldf in iso case


!!gm  Question:  here is it really Kmm that should be used and not Kbb  ?


!!gm BUG ?:  below akz calculation should use zmsku/v instead of * 0.25_wp
!!gm        ===>>>  introduce ah_wspl2 calculation in all cases !

      IF( ln_traldf_msc ) THEN               ! stabilizing vertical diffusivity coefficient (compute akz)
         DO_3D( inn , inn , inn , inn ,   2, jpkm1 )
            !                                ! round brackets required to ensure halo size compatibility with north fold boundary condition
            pakz(ji,jj,jk) = (   (  ( ahtu(ji  ,jj,jk) + ahtu(ji  ,jj,jk-1) ) / ( e1u(ji  ,jj) * e1u(ji  ,jj) )      &
               &                  + ( ahtu(ji-1,jj,jk) + ahtu(ji-1,jj,jk-1) ) / ( e1u(ji-1,jj) * e1u(ji-1,jj) )  )   &
               &               + (  ( ahtv(ji,jj  ,jk) + ahtv(ji,jj  ,jk-1) ) / ( e2v(ji,jj  ) * e2v(ji,jj  ) )      &
               &                  + ( ahtv(ji,jj-1,jk) + ahtv(ji,jj-1,jk-1) ) / ( e2v(ji,jj-1) * e2v(ji,jj-1) )  )   ) * 0.25_wp
         END_3D
         !
         IF( ln_traldf_blp ) THEN            ! bilaplacian operator
            DO_3D( inn , inn , inn , inn , 2, jpkm1 )
               ze3w_2 = e3w(ji,jj,jk,Kmm) * e3w(ji,jj,jk,Kmm)
               pakz(ji,jj,jk) = 16._wp * pah_wslp2(ji,jj,jk) * (  pakz(ji,jj,jk) + pah_wslp2(ji,jj,jk) / ze3w_2  )
            END_3D
         ELSEIF( ln_traldf_lap ) THEN        ! laplacian operator
            DO_3D( inn , inn , inn , inn ,   2, jpkm1 )
               ze3w_2 = e3w(ji,jj,jk,Kmm) * e3w(ji,jj,jk,Kmm)
               zcoef0 = rDt * (  pakz(ji,jj,jk) + pah_wslp2(ji,jj,jk) / ze3w_2  )
               pakz(ji,jj,jk) = MAX( zcoef0 - 0.5_wp , 0._wp ) * ze3w_2 * r1_Dt
            END_3D
        ENDIF
        !
      ELSE                                    ! A33 flux set to zero with akz=ah_wslp2 ==>> computed in full implicit
         DO_3D( inn , inn , inn , inn ,   1, jpk )
            pakz(ji,jj,jk) = pah_wslp2(ji,jj,jk)
         END_3D
      ENDIF
      !
   END SUBROUTINE traldf_iso_a33

   !!==============================================================================
END MODULE traldf_iso
