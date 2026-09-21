MODULE divhor
   !!==============================================================================
   !!                       ***  MODULE  divhor  ***
   !! Ocean diagnostic variable : now horizontal divergence
   !!==============================================================================
   !! History :  1.0  ! 2002-09  (G. Madec, E. Durand)  Free form, F90
   !!             -   ! 2005-01  (J. Chanut) Unstructured open boundaries
   !!             -   ! 2003-08  (G. Madec)  merged of cur and div, free form, F90
   !!             -   ! 2005-01  (J. Chanut, A. Sellar) unstructured open boundaries
   !!            3.3  ! 2010-09  (D.Storkey and E.O'Dea) bug fixes for BDY module
   !!             -   ! 2010-10  (R. Furner, G. Madec) runoff and cla added directly here
   !!            3.7  ! 2014-01  (G. Madec) suppression of velocity curl from in-core memory
   !!             -   ! 2014-12  (G. Madec) suppression of cross land advection option
   !!             -   ! 2015-10  (G. Madec) add velocity and rnf flag in argument of div_hor
   !!            4.5  ! 2015-10  (S. Techene, G. Madec)  hdiv replaced by e3divUh
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   div_hor       : Compute the horizontal divergence field
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and tracers
   USE dom_oce        ! ocean space and time domain
   USE domutl  , ONLY : lbnd_ij
   USE sbc_oce , ONLY : ln_rnf      ! river runoff
   USE sbcrnf  , ONLY : sbc_rnf_div ! river runoff 
   USE isf_oce , ONLY : ln_isf      ! ice shelf
   USE isfhdiv , ONLY : isf_hdiv    ! ice shelf
#if defined key_asminc   
   USE asminc         ! Assimilation increment
#endif
   !
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! MPP library
   USE timing         ! Timing

   IMPLICIT NONE
   PRIVATE

   !! * Interface
   INTERFACE div_hor
      MODULE PROCEDURE div_hor, div_hor_old
   END INTERFACE

   PUBLIC   div_hor   ! routine called by sshwzv.F90  : ssh_nxt(div_hor_old) and wzv(div_hor)

   INTEGER, PUBLIC, PARAMETER ::   np_velocity  = 0   ! velocities given as arguments in wzv
   INTEGER, PUBLIC, PARAMETER ::   np_transport = 1   ! transports given as arguments in wzv

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE div_hor( kt, Kbb, Kmm, pu, pv, pe3divUh, k_ind )
      !!
      INTEGER                        , INTENT(in   ) ::   kt, Kbb, Kmm   ! ocean time-step & time-level indices
      INTEGER , OPTIONAL             , INTENT(in   ) ::   k_ind          ! indicator
      REAL(wp), DIMENSION(:,:,:)     , INTENT(in   ) ::   pu, pv         ! horizontal velocity or transport
      REAL(wp), DIMENSION(T2D(1),jpk), INTENT(  out) ::   pe3divUh       ! e3t*div[Uh]
      !!
      CALL div_hor_t( kt, Kbb, Kmm, pu, pv, lbnd_ij(pu), pe3divUh, k_ind )
   END SUBROUTINE div_hor


   SUBROUTINE div_hor_t( kt, Kbb, Kmm, pu, pv, ktuv, pe3divUh, k_ind )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE div_hor  ***
      !!                    
      !! ** Purpose :   compute the horizontal divergence at now time-step
      !!
      !! ** Method  :   the now divergence is computed as :
      !!         hdiv = 1/(e1e2t*e3t) ( di[e2u*e3u un] + dj[e1v*e3v vn] )
      !!      and correct with runoff inflow (div_rnf) and cross land flow (div_cla) 
      !!
      !! ** Action  : - thickness weighted horizontal divergence of in input velocity (puu,pvv)
      !!----------------------------------------------------------------------
      INTEGER,  DIMENSION(2)             , INTENT(in   ) ::   ktuv
      INTEGER                            , INTENT(in   ) ::   kt, Kbb, Kmm   ! ocean time-step & time-level indices
      INTEGER , OPTIONAL                 , INTENT(in   ) ::   k_ind          ! indicator
      REAL(wp), DIMENSION(AB2D(ktuv),JPK), INTENT(in   ) ::   pu, pv         ! horizontal velocity or transport
      REAL(wp), DIMENSION(T2D(1),jpk)    , INTENT(  out) ::   pe3divUh       ! e3t*div[Uh]
      !
      INTEGER  ::   ji, jj, jk    ! dummy loop indices
      INTEGER  ::   ik_ind        ! local indicator
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('div_hor')
      !
      IF( kt == nit000 ) THEN
         IF( .NOT. l_istiled .OR. ntile == 1 )  THEN                       ! Do only on the first tile
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) 'div_hor : thickness weighted horizontal divergence '
            IF(lwp) WRITE(numout,*) '~~~~~~~~~~~'
         ENDIF
      ENDIF
      !
      IF( .NOT. PRESENT( k_ind ) ) THEN
         ik_ind = np_velocity
      ELSE
         ik_ind = k_ind
      ENDIF
      !
      SELECT CASE ( ik_ind )
      CASE ( np_velocity )
         DO_3D( 1, 1, 1, 1, 1, jpkm1 )
            hdiv(ji,jj,jk) = (  (  e2u(ji  ,jj) * e3u(ji  ,jj,jk,Kmm) * pu(ji  ,jj,jk)       &   ! add () for NP repro
               &                 - e2u(ji-1,jj) * e3u(ji-1,jj,jk,Kmm) * pu(ji-1,jj,jk) )     &
               &              + (  e1v(ji,jj  ) * e3v(ji,jj  ,jk,Kmm) * pv(ji,jj  ,jk)       &
               &                 - e1v(ji,jj-1) * e3v(ji,jj-1,jk,Kmm) * pv(ji,jj-1,jk) )     &
               &             ) * r1_e1e2t(ji,jj) / e3t(ji,jj,jk,Kmm)
         END_3D
      CASE ( np_transport )
         DO_3D( 1, 1, 1, 1, 1, jpkm1 )
            hdiv(ji,jj,jk) = (  (  pu(ji  ,jj,jk)       &   ! add () for NP repro
               &                 - pu(ji-1,jj,jk) )     &
               &              + (  pv(ji,jj  ,jk)       &
               &                 - pv(ji,jj-1,jk) )     &
               &             ) * r1_e1e2t(ji,jj) / e3t(ji,jj,jk,Kmm)
         END_3D
      END SELECT
      !
      IF( ln_rnf )   CALL sbc_rnf_div( hdiv, Kmm )             !==  + runoffs divergence  ==!
      !
#if defined key_asminc 
      IF( ln_sshinc .AND. ln_asmiau )   &                      !==  + SSH assimilation increment  ==!
         &           CALL ssh_asm_div( kt, Kbb, Kmm, hdiv )
#endif
      !
      IF( ln_isf )   CALL isf_hdiv( kt, Kmm, hdiv )            !==  + ice-shelf mass exchange ==!
      !
!!gm Patch before suppression of hdiv from all modules that use it
!      DO_3D( 0, 0, 0, 0, 1, jpkm1 )                            !==  e3t * Horizontal divergence  ==!
!         pe3divUh(ji,jj,jk) = hdiv(ji,jj,jk) * e3t(ji,jj,jk,Kmm)
!      END_3D
!JC: over whole domain, and after lbclnk on hdiv to prevent from reproducibility issues
      DO_3D( 1, 1, 1, 1, 1, jpkm1 )
         pe3divUh(ji,jj,jk) = hdiv(ji,jj,jk) * e3t(ji,jj,jk,Kmm)
      END_3D
!!gm end
      !
      !
      IF( ln_timing )   CALL timing_stop('div_hor')
      !
   END SUBROUTINE div_hor_t


   SUBROUTINE div_hor_old( kt, Kbb, Kmm )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE div_hor  ***
      !!                    
      !! ** Purpose :   compute the horizontal divergence at now time-step
      !!
      !! ** Method  :   the now divergence is computed as :
      !!         hdiv = 1/(e1e2t*e3t) ( di[e2u*e3u un] + dj[e1v*e3v vn] )
      !!      and correct with runoff inflow (div_rnf) and cross land flow (div_cla) 
      !!
      !! ** Action  : - update hdiv, the now horizontal divergence
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt        ! ocean time-step index
      INTEGER, INTENT(in) ::   Kbb, Kmm  ! ocean time level indices
      !
      INTEGER  ::   ji, jj, jk    ! dummy loop indices
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('div_hor')
      !
      IF( kt == nit000 ) THEN
         IF( .NOT. l_istiled .OR. ntile == 1 )  THEN                       ! Do only on the first tile
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) 'div_hor : horizontal velocity divergence '
            IF(lwp) WRITE(numout,*) '~~~~~~~   '
         ENDIF
      ENDIF
      !
      DO_3D( 1, 1, 1, 1, 1, jpkm1 )                                          !==  Horizontal divergence  ==!
         hdiv(ji,jj,jk) = (  (  e2u(ji  ,jj) * e3u(ji  ,jj,jk,Kmm) * uu(ji  ,jj,jk,Kmm)     &   ! add () for NP repro
            &                 - e2u(ji-1,jj) * e3u(ji-1,jj,jk,Kmm) * uu(ji-1,jj,jk,Kmm) )   &
            &              + (  e1v(ji,jj  ) * e3v(ji,jj  ,jk,Kmm) * vv(ji,jj  ,jk,Kmm)     &
            &                 - e1v(ji,jj-1) * e3v(ji,jj-1,jk,Kmm) * vv(ji,jj-1,jk,Kmm) )   &
            &             ) * r1_e1e2t(ji,jj) / e3t(ji,jj,jk,Kmm)
      END_3D
      !
      IF( ln_rnf )   CALL sbc_rnf_div( hdiv, Kmm )                               !==  runoffs    ==!   (update hdiv field)
      !
#if defined key_asminc 
      IF( ln_sshinc .AND. ln_asmiau )   CALL ssh_asm_div( kt, Kbb, Kmm, hdiv )   !==  SSH assimilation  ==!   (update hdiv field)
      ! 
#endif
      IF( ln_isf )   CALL isf_hdiv( kt, Kmm, hdiv )                              !==  ice shelf  ==!   (update hdiv field)
      !
      IF( ln_timing )   CALL timing_stop('div_hor')
      !
   END SUBROUTINE div_hor_old
   
   !!======================================================================
END MODULE divhor
