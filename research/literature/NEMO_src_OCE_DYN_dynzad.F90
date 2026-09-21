MODULE dynzad
   !!======================================================================
   !!                       ***  MODULE  dynzad  ***
   !! Ocean dynamics : vertical advection trend
   !!======================================================================
   !! History :  OPA  !  1991-01  (G. Madec) Original code
   !!   NEMO     0.5  !  2002-07  (G. Madec) Free form, F90
   !!            4.5  !  2021-01  (S. Techene, G. Madec) memory optimization
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dyn_zad       : vertical advection momentum trend
   !!----------------------------------------------------------------------
   USE oce            ! ocean dynamics and tracers
   USE dom_oce        ! ocean space and time domain
   USE sbc_oce        ! surface boundary condition: ocean
   USE trd_oce        ! trends: ocean variables
   USE trddyn         ! trend manager: dynamics
   USE sbcwave, ONLY: wsd   ! Surface Waves (add vertical Stokes-drift)
   !
   USE in_out_manager ! I/O manager
   USE lib_mpp        ! MPP library
   USE prtctl         ! Print control
   USE timing         ! Timing

   IMPLICIT NONE
   PRIVATE

   PUBLIC   dyn_zad       ! routine called by stp2d.F90 and dynadv.F90

   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE dyn_zad ( kt, Kmm, puu, pvv, Krhs )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dynzad  ***
      !!
      !! ** Purpose :   Compute the now vertical momentum advection trend and
      !!      add it to the general trend of momentum equation.
      !!
      !! ** Method  :   The now vertical advection of momentum is given by:
      !!         w dz(u) = u(rhs) + 1/(e1e2u*e3u) mk+1[ mi(e1e2t*ww) dk(u) ]
      !!         w dz(v) = v(rhs) + 1/(e1e2v*e3v) mk+1[ mj(e1e2t*ww) dk(v) ]
      !!      Add this trend to the general trend (puu(:,:,:,Krhs),pvv(:,:,:,Krhs)):
      !!         (u(rhs),v(rhs)) = (u(rhs),v(rhs)) + w dz(u,v)
      !!
      !! ** Action  : - Update (puu(:,:,:,Krhs),pvv(:,:,:,Krhs)) with the vert. momentum adv. trends
      !!              - Send the trends to trddyn for diagnostics (l_trddyn=T)
      !!----------------------------------------------------------------------
      INTEGER                             , INTENT(in   ) ::    kt, Kmm, Krhs   ! ocean time-step & time-level indices
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(inout) ::    puu, pvv        ! ocean velocities and RHS of momentum equation
      !
      INTEGER  ::   ji, jj, jk   ! dummy loop indices
      REAL(wp) ::   zWf, zWfi, zzWfu, zzWdzU   ! local scalars
      REAL(wp) ::        zWfj, zzWfv, zzWdzV   !   -      -
      REAL(wp), DIMENSION(T2D(0))             ::   zWdzU, zWdzV   ! 2D inner workspace
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   ztrdu, ztrdv   ! 3D workspace
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('dyn_zad')
      !
      IF( .NOT. l_istiled .OR. ntile == 1 )  THEN                       ! Do only on the first tile
         IF( kt == nit000 ) THEN
            IF(lwp) WRITE(numout,*)
            IF(lwp) WRITE(numout,*) 'dyn_zad : 2nd order vertical advection scheme'
         ENDIF
      ENDIF
      !
      IF( l_trddyn )   THEN           ! Save puu(:,:,:,Krhs) and pvv(:,:,:,Krhs) trends
         ALLOCATE( ztrdu(T2D(0),jpk), ztrdv(T2D(0),jpk) )
         ztrdu(:,:,:) = puu(T2D(0),:,Krhs)
         ztrdv(:,:,:) = pvv(T2D(0),:,Krhs)
      ENDIF
      !
      !                                !==  vertical momentum advection ==!   at u- and v-points
      !
      zWdzU(T2D(0)) = 0._wp                  ! set surface (jk=1) vertical advection to zero
      zWdzV(T2D(0)) = 0._wp
      !        
      DO_3D( 0, 0, 0, 0 , 1, jpk-2 )   !=  surface to jpk-2 vertical advection
         !                                ! vertical transport at jk+1 uw/vw-level (x2): 2*mi/j[e1e2t*(We)]
         IF( ln_vortex_force ) THEN             ! We = ww+wsd
            zWf  = e1e2t(ji  ,jj  ) * ( ww(ji  ,jj  ,jk+1) + wsd(ji  ,jj  ,jk+1) )
            zWfi = e1e2t(ji+1,jj  ) * ( ww(ji+1,jj  ,jk+1) + wsd(ji+1,jj  ,jk+1) )
            zWfj = e1e2t(ji  ,jj+1) * ( ww(ji  ,jj+1,jk+1) + wsd(ji  ,jj+1,jk+1) )
         ELSE                                   ! We = ww
            zWf  = e1e2t(ji  ,jj  ) *   ww(ji  ,jj  ,jk+1)
            zWfi = e1e2t(ji+1,jj  ) *   ww(ji+1,jj  ,jk+1)
            zWfj = e1e2t(ji  ,jj+1) *   ww(ji  ,jj+1,jk+1)
         ENDIF
         zzWfu = zWfi + zWf                     ! averaging at uw- and vw-points (x2)
         zzWfv = zWfj + zWf 
         !                                ! vertical advection at jk+1 uw-level (x4): zzWfu/v*dk+1[u/v] 
         zzWdzU = zzWfu * ( puu(ji,jj,jk,Kmm) - puu(ji,jj,jk+1,Kmm) )
         zzWdzV = zzWfv * ( pvv(ji,jj,jk,Kmm) - pvv(ji,jj,jk+1,Kmm) )
         !
         !                                ! vertical advection  at jk u/v-level 
         puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp * r1_e1e2u(ji,jj) / e3u(ji,jj,jk,Kmm)   &
              &                                            * ( zWdzU(ji,jj) + zzWdzU )
         pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - 0.25_wp * r1_e1e2v(ji,jj) / e3v(ji,jj,jk,Kmm)   &
              &                                            * ( zWdzV(ji,jj) + zzWdzV )
         !
         zWdzU(ji,jj) = zzWdzU            ! save for next level computation 
         zWdzV(ji,jj) = zzWdzV
      END_3D
      !
      jk = jpkm1
      DO_2D( 0, 0, 0, 0 )           !=  bottom vertical advection  at jpkm1
         puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp * r1_e1e2u(ji,jj) / e3u(ji,jj,jk,Kmm)   &
            &                                              * zWdzU(ji,jj)
         pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - 0.25_wp * r1_e1e2v(ji,jj) / e3v(ji,jj,jk,Kmm)   &
            &                                              * zWdzV(ji,jj)
      END_2D
      !
      IF( l_trddyn ) THEN             ! save the vertical advection trends for diagnostic
         ztrdu(:,:,:) = puu(T2D(0),:,Krhs) - ztrdu(:,:,:)
         ztrdv(:,:,:) = pvv(T2D(0),:,Krhs) - ztrdv(:,:,:)
         CALL trd_dyn( ztrdu, ztrdv, jpdyn_zad, kt, Kmm )
         DEALLOCATE( ztrdu, ztrdv )
      ENDIF
      !                               ! Control print
      IF(sn_cfctl%l_prtctl)   CALL prt_ctl( tab3d_1=puu(:,:,:,Krhs), clinfo1=' zad  - Ua: ', mask1=umask,   &
         &                                  tab3d_2=pvv(:,:,:,Krhs), clinfo2=       ' Va: ', mask2=vmask, clinfo3='dyn' )
      !
      IF( ln_timing )   CALL timing_stop('dyn_zad')
      !
   END SUBROUTINE dyn_zad

   !!======================================================================
END MODULE dynzad
