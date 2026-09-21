MODULE domzgr
   !!==============================================================================
   !!                       ***  MODULE domzgr   ***
   !! Ocean domain : definition of the vertical coordinate system
   !!==============================================================================
   !! History :  OPA  ! 1995-12  (G. Madec)  Original code : s vertical coordinate
   !!                 ! 1997-07  (G. Madec)  lbc_lnk call
   !!                 ! 1997-04  (J.-O. Beismann) 
   !!            8.5  ! 2002-09  (A. Bozec, G. Madec)  F90: Free form and module
   !!             -   ! 2002-09  (A. de Miranda)  rigid-lid + islands
   !!  NEMO      1.0  ! 2003-08  (G. Madec)  F90: Free form and module
   !!             -   ! 2005-10  (A. Beckmann)  modifications for hybrid s-ccordinates & new stretching function
   !!            2.0  ! 2006-04  (R. Benshila, G. Madec)  add zgr_zco
   !!            3.0  ! 2008-06  (G. Madec)  insertion of domzgr_zps.h90 & conding style
   !!            3.2  ! 2009-07  (R. Benshila) Suppression of rigid-lid option
   !!            3.3  ! 2010-11  (G. Madec) add mbk. arrays associated to the deepest ocean level
   !!            3.4  ! 2012-08  (J. Siddorn) added Siddorn and Furner stretching function
   !!            3.4  ! 2012-12  (R. Bourdalle-Badie and G. Reffray)  modify C1D case  
   !!            3.6  ! 2014-11  (P. Mathiot and C. Harris) add ice shelf capabilitye  
   !!            3.?  ! 2015-11  (H. Liu) Modifications for Wetting/Drying
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dom_zgr       : read or set the ocean vertical coordinate system
   !!   zgr_top_bot   : ocean top and bottom level for t-, u, and v-points with 1 as minimum value
   !!---------------------------------------------------------------------
   USE oce            ! ocean variables
   USE dom_oce        ! ocean domain
   USE usrdef_zgr     ! user defined vertical coordinate system
   USE closea         ! closed seas
   USE depth_e3       ! depth <=> e3
   USE wet_dry,   ONLY: ll_wd, ssh_ref  ! Wetting and drying
   !
   USE in_out_manager ! I/O manager
   USE iom            ! I/O library
   USE lbclnk         ! ocean lateral boundary conditions (or mpp link)
   USE lib_mpp        ! distributed memory computing library

   IMPLICIT NONE
   PRIVATE

   PUBLIC   dom_zgr        ! called by dom_init.F90

  !! * Substitutions
#  include "do_loop_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS       

   SUBROUTINE dom_zgr( k_top, k_bot )
      !!----------------------------------------------------------------------
      !!                ***  ROUTINE dom_zgr  ***
      !!                   
      !! ** Purpose :   set the depth of model levels and the resulting 
      !!              vertical scale factors.
      !!
      !! ** Method  : - reference 1D vertical coordinate (gdep._1d, e3._1d)
      !!              - read/set ocean depth and ocean levels (bathy, mbathy)
      !!              - vertical coordinate (gdep., e3.) depending on the 
      !!                coordinate chosen :
      !!                   l_zco=T   z-coordinate   
      !!                   l_zps=T   z-coordinate with partial steps
      !!                   l_zco=T   s-coordinate 
      !!
      !! ** Action  :   define gdep., e3., mbathy and bathy
      !!----------------------------------------------------------------------
      INTEGER, DIMENSION(:,:), INTENT(out) ::   k_top, k_bot   ! ocean first and last level indices
      !
      INTEGER  ::   ji,jj,jk            ! dummy loop index
      INTEGER  ::   ikt, ikb            ! top/bot index
      INTEGER  ::   ioptio, inum, iatt  ! local integer
      INTEGER  ::   is_mbkuvf           ! ==0 if mbku, mbkv, mbkf to be computed
      REAL(wp) ::   zrefdep             ! depth of the reference level (~10m)
      REAL(WP) ::   z_zco, z_zps, z_sco, z_cav
      CHARACTER(len=7) ::   catt        ! 'zco', 'zps, 'sco' or 'UNKNOWN'
      REAL(wp), DIMENSION(jpi,jpj)   ::   zmsk, z2d
      REAL(wp), DIMENSION(jpi,jpj,2) ::   ztopbot
      !!----------------------------------------------------------------------
      !
      IF(lwp) THEN                     ! Control print
         WRITE(numout,*)
         WRITE(numout,*) 'dom_zgr : vertical coordinate'
         WRITE(numout,*) '~~~~~~~'
         IF( lk_linssh ) WRITE(numout,*) '          linear free surface: the vertical mesh does not change in time'
      ENDIF
      !                                ! Control keys
      IF(.NOT.( lk_linssh.OR.lk_qco ) .OR. ( lk_linssh.AND.lk_qco ) )   &
         &              CALL ctl_stop( 'STOP','domzgr: either key_linssh or key_qco MUST be set up in cpp_* file !' )
      !
      IF(.NOT.lk_vco_1d .AND. .NOT.lk_vco_1d3d .AND. .NOT.lk_vco_3d )   &
         &              CALL ctl_stop( 'STOP','domzgr: either key_vco_1d, key_vco_1d3d or key_vco_3d MUST be set up in cpp_* file !' )
      !
      !                             !==============================!
      IF( ln_read_cfg ) THEN        !==  read in domcfg.nc file  ==!
         !                          !==============================!
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) '   ==>>>   Read vertical mesh in ', TRIM( cn_domcfg ), ' file'
         is_mbkuvf = 0         
         !
         CALL iom_open( cn_domcfg, inum )    ! open domcfg file
         !
!!st set ln_isfcav
         IF( lk_isf ) THEN 
            CALL iom_getatt( inum,    'IsfCav', iatt )   ! returns -999 if not found
            ln_isfcav = iatt == 1           ! default = .false.
            !
            ! ------- keep compatibility with OLD VERSION... start -------
            IF( iatt == -999 ) THEN
               CALL iom_get( inum, 'ln_isfcav', z_cav )   ;   ln_isfcav = z_cav /= 0._wp
            ENDIF
         ENDIF
         !                                         !* type of vertical coordinate
         CALL iom_getatt( inum, 'VertCoord', catt )   ! returns 'UNKNOWN' if not found
         l_zco = catt == 'zco'          ! default = .false.
         l_zps = catt == 'zps'          ! default = .false.
         l_sco = catt == 'sco'          ! default = .false.
         !
         ! ------- keep compatibility with OLD VERSION... start -------
         IF( catt == 'UNKNOWN' ) THEN
            CALL iom_get( inum,    'ln_zco', z_zco )   ;   l_zco = z_zco /= 0._wp
            CALL iom_get( inum,    'ln_zps', z_zps )   ;   l_zps = z_zps /= 0._wp
            CALL iom_get( inum,    'ln_sco', z_sco )   ;   l_sco = z_sco /= 0._wp
         ENDIF
         !                                !------------------------------!
         !                                !--  all coordinate systems  --!   1D depth and e3   (needed in netcdf files)
         !                                !------------------------------!
         !
         CALL iom_get( inum, jpdom_unknown, 'e3t_1d'  , e3t_1d  )   ! 1D reference coordinate
         CALL iom_get( inum, jpdom_unknown, 'e3w_1d'  , e3w_1d  )
         CALL e3_to_depth( e3t_1d, e3w_1d, gdept_1d, gdepw_1d )     ! 1D reference depth deduced from e3
!!st NEED to deduce e3 from gdep 
!!st         CALL iom_get( inum, jpdom_unknown, 'gdept_1d', gdept_1d )   ! 1D depth read
!!st         CALL iom_get( inum, jpdom_unknown, 'gdepw_1d', gdepw_1d )
!!st         CALL depth_to_e3( gdept_1d, gdepw_1d, e3t_1d, e3w_1d )      ! 1D e3 deduced from depth
         !
         !                                   !- ocean top and bottom k-indices
         CALL iom_get( inum, jpdom_global, 'top_level'    , z2d(:,:)   )   ! 1st wet T-points (ISF)
         k_top(:,:) = NINT( z2d(:,:) )
         CALL iom_get( inum, jpdom_global, 'bottom_level' , z2d(:,:)   )   ! last wet T-points
         k_bot(:,:) = NINT( z2d(:,:) )
         !
         IF( iom_varid( inum, 'mbku', ldstop = .FALSE. ) > 0 ) THEN
            IF(lwp) WRITE(numout,*) '          mbku, mbkv & mbkf read in ', TRIM(cn_domcfg), ' file'
            CALL iom_get( inum, jpdom_global, 'mbku', z2d, cd_type = 'U', psgn = 1._wp, kfill = jpfillcopy )
            mbku(:,:) = NINT( z2d(:,:) )
            CALL iom_get( inum, jpdom_global, 'mbkv', z2d, cd_type = 'V', psgn = 1._wp, kfill = jpfillcopy )
            mbkv(:,:) = NINT( z2d(:,:) )
            CALL iom_get( inum, jpdom_global, 'mbkf', z2d, cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )
            mbkf(:,:) = NINT( z2d(:,:) )
            is_mbkuvf = 1
         ELSE
            is_mbkuvf = 0
         ENDIF
         !
         !                                !--------------------!
         IF(     lk_vco_1d ) THEN         !--  z-coordinate  --!   use only 1D arrays for all gdep and e3 fields
            !                             !--------------------!
            !
            IF(.NOT. l_zco )   CALL ctl_stop( 'STOP','domzgr: only l_zco is compatible with key_vco_1d. Fix domcfg or cpp_* file !' )
            !
            !                             !-----------------------!
         ELSEIF( lk_vco_1d3d ) THEN       !--  z-partial cells  --!   use 3D t-level e3
            !                             !-----------------------!
            !
            IF( l_sco )   CALL ctl_stop( 'STOP','domzgr: l_sco is NOT compatible with key_vco_1d3d. Fix domcfg or cpp_* file !' )
            !
            !                                   ! t-level: 3D reference   (include partial cell)
            CALL iom_get( inum, jpdom_global, 'e3t_0'  , e3t_3d, cd_type = 'T', psgn = 1._wp, kfill = jpfillcopy )
            CALL iom_get( inum, jpdom_global, 'e3u_0'  , e3u_3d, cd_type = 'U', psgn = 1._wp, kfill = jpfillcopy )
            CALL iom_get( inum, jpdom_global, 'e3v_0'  , e3v_3d, cd_type = 'V', psgn = 1._wp, kfill = jpfillcopy )
            CALL iom_get( inum, jpdom_global, 'e3f_0'  , e3f_3d, cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )
            !
            !                             !--------------------!               
         ELSEIF( lk_vco_3d ) THEN         !--  s-coordinate  --!   use 3D for all gdep and e3 fields
            !                             !--------------------!
            !
            !                                   !* depth         : 3D reference   (without partial cell)
!!st no gdep._0 in ORCA_domcfg.nc from sette
!!st            CALL iom_get( inum, jpdom_global , 'gdept_0' , gdept_0, kfill = jpfillcopy )
!!st            CALL iom_get( inum, jpdom_global , 'gdepw_0' , gdepw_0, kfill = jpfillcopy )
            !
            !                                   !* scale factors : 3D reference   (can include partial cell)
            CALL iom_get( inum, jpdom_global, 'e3t_0'  , e3t_3d , cd_type = 'T', psgn = 1._wp, kfill = jpfillcopy )
            CALL iom_get( inum, jpdom_global, 'e3u_0'  , e3u_3d , cd_type = 'U', psgn = 1._wp, kfill = jpfillcopy )
            CALL iom_get( inum, jpdom_global, 'e3v_0'  , e3v_3d , cd_type = 'V', psgn = 1._wp, kfill = jpfillcopy )
            CALL iom_get( inum, jpdom_global, 'e3f_0'  , e3f_3d , cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )
            CALL iom_get( inum, jpdom_global, 'e3w_0'  , e3w_3d , cd_type = 'W', psgn = 1._wp, kfill = jpfillcopy )
            CALL iom_get( inum, jpdom_global, 'e3uw_0' , e3uw_3d, cd_type = 'U', psgn = 1._wp, kfill = jpfillcopy )
            CALL iom_get( inum, jpdom_global, 'e3vw_0' , e3vw_3d, cd_type = 'V', psgn = 1._wp, kfill = jpfillcopy )
            !
!!st PATCH for ORCA see above no gdep._0 in ORCA_domcfg.nc
            IF(  iom_varid( inum, 'gdept_0' , ldstop = .FALSE. ) > 0  .AND.  &
                 & iom_varid( inum, 'gdepw_0' , ldstop = .FALSE. ) > 0    ) THEN
               CALL ctl_warn( 'zgr_read : old definition of depths and scale factors used ', & 
                    &           '           depths at t- and w-points read in the domain configuration file')
               CALL iom_get( inum, jpdom_global , 'gdept_0' , gdept_3d, kfill = jpfillcopy )
               CALL iom_get( inum, jpdom_global , 'gdepw_0' , gdepw_3d, kfill = jpfillcopy )
               !
            ELSE                                !- depths computed from e3. scale factors
               CALL e3_to_depth( e3t_3d   , e3w_3d   , gdept_3d   , gdepw_3d    )    ! 3D depths
            ENDIF

!!st            IF(l_zps) THEN
!!st               DO jk=1, jpk
!!st                   e3w_3d(:,:,jk) = e3w_1d(jk)
!!st                  e3uw_3d(:,:,jk) = e3w_1d(jk)
!!st                  e3vw_3d(:,:,jk) = e3w_1d(jk)
!!st                  gdept_3d(:,:,jk) = gdept_1d(jk)
!!st                  gdepw_3d(:,:,jk) = gdepw_1d(jk)
!!st               END DO  
!!st            ENDIF
            !                                   !* reference depth for negative bathy (wetting and drying only)
            IF( ll_wd )   CALL iom_get( inum,  'rn_wd_ref_depth' , ssh_ref   )
            !
            !                             !----------------------!               
         ELSEIF( lk_ALE ) THEN            !--  ALE-coordinate  --!   combine time & space variations
            !                             !----------------------!               
            !!gm 
            !    to be done : no restart read the 3D ref coord in domcfg to set coordinate at Nbb  
            !               :    restart read the 3D ref coord when restart at Nbb 
            !
            ! Initialisation only (NO restart)
            !                                   !* depth         :  4D fields   (without partial cell)
!             CALL iom_get( inum, jpdom_global , 'gdept' , gdept(:,:,:,Kbb), kfill = jpfillcopy )
!             CALL iom_get( inum, jpdom_global , 'gdepw' , gdepw, kfill = jpfillcopy )
!             !
!             !                                 !* scale factors :  4D fields   (can include partial cell)
!             CALL iom_get( inum, jpdom_global, 'e3t_0'  ,  e3t(:,:,:,Kbb), cd_type = 'T', psgn = 1._wp, kfill = jpfillcopy )
!             CALL iom_get( inum, jpdom_global, 'e3u_0'  ,  e3u(:,:,:,Kbb), cd_type = 'U', psgn = 1._wp, kfill = jpfillcopy )
!             CALL iom_get( inum, jpdom_global, 'e3v_0'  ,  e3v(:,:,:,Kbb), cd_type = 'V', psgn = 1._wp, kfill = jpfillcopy )
!             CALL iom_get( inum, jpdom_global, 'e3f_0'  ,  e3f(:,:,:,Kbb), cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )
!             CALL iom_get( inum, jpdom_global, 'e3w_0'  ,  e3w(:,:,:,Kbb), cd_type = 'W', psgn = 1._wp, kfill = jpfillcopy )
!             CALL iom_get( inum, jpdom_global, 'e3uw_0' , e3uw(:,:,:,Kbb), cd_type = 'U', psgn = 1._wp, kfill = jpfillcopy )
!             CALL iom_get( inum, jpdom_global, 'e3vw_0' , e3vw(:,:,:,Kbb), cd_type = 'V', psgn = 1._wp, kfill = jpfillcopy )
            !!gm
         ENDIF
         !
         CALL iom_close( inum )        ! close domcfg file
         !
         !                          !==================================!
      ELSE                          !==  User defined configuration  ==!
         !                          !==================================!
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) '          User defined vertical mesh (usr_def_zgr)'
         is_mbkuvf = 0
         !
         !                                !--------------------!
         IF( lk_vco_1d ) THEN             !--  z-coordinate  --!   use only 1D arrays for all gdep and e3 fields
            !                             !--------------------!
            !
            
            CALL usr_def_zgr( l_zco  , l_zps  , l_sco, ln_isfcav,   &
                 &              k_top   , k_bot                      ,   &    ! 1st & last ocean level
                 &              gdept_1d, gdepw_1d, e3t_1d, e3w_1d   )        ! 1D gridpoints depth
            !
            IF( l_sco )   CALL ctl_stop( 'STOP','domzgr: key_vco_1d and l_sco=T are incompatible. Fix usrdef_zgr !' )
            IF( l_zps )   CALL ctl_stop( 'STOP','domzgr: key_vco_1d and l_zps=T are incompatible. Fix usrdef_zgr !' )
            !
            !                             !-----------------------!
         ELSEIF( lk_vco_1d3d ) THEN       !--  z-partial cells  --!   use 3D t-level e3
            !                             !-----------------------!
            !
            CALL usr_def_zgr( l_zco  , l_zps  , l_sco, ln_isfcav   ,   &
                 &              k_top   , k_bot                      ,   &    ! 1st & last ocean level
                 &              gdept_1d, gdepw_1d, e3t_1d, e3w_1d   ,   &    ! 1D gridpoints depth
                 &              e3t_3d  , e3u_3d  , e3v_3d, e3f_3d       )    ! vertical scale factors
            !
            IF( l_sco )   CALL ctl_stop( 'STOP','domzgr: key_vco_1d3d and l_sco=T are incompatible. Fix usrdef_zgr !' )
            !
            ! make sure that periodicities are properly applied 
            CALL lbc_lnk( 'dom_zgr', e3t_3d, 'T', 1._wp,   e3u_3d, 'U', 1._wp,  e3v_3d, 'V', 1._wp, e3f_3d, 'F', 1._wp,   &
                 &                     kfillmode = jpfillcopy )   ! do not put 0 over closed boundaries
            !
            !                             !--------------------!               
         ELSEIF( lk_vco_3d ) THEN         !--  s-coordinate  --!   use 3D for all gdep and e3 fields
            !                             !--------------------!
            !
            CALL usr_def_zgr( l_zco  , l_zps  , l_sco, ln_isfcav   ,   &
                 &              k_top   , k_bot                      ,   &    ! 1st & last ocean level
                 &              gdept_1d, gdepw_1d, e3t_1d, e3w_1d   ,   &    ! 1D gridpoints depth
                 &              e3t_3d  , e3u_3d  , e3v_3d, e3f_3d   ,   &    ! vertical scale factors
                 &              gdept_3d, gdepw_3d                   ,   &    ! gridpoints depth 
                 &              e3w_3d  , e3uw_3d , e3vw_3d               )   ! vertical scale factors
            CALL lbc_lnk( 'dom_zgr', gdept_3d, 'T', 1._wp, gdepw_3d, 'W', 1._wp,                                            &
                 &                       e3t_3d, 'T', 1._wp,   e3u_3d, 'U', 1._wp,  e3v_3d, 'V', 1._wp, e3f_3d, 'F', 1._wp,   &
                 &                       e3w_3d, 'W', 1._wp,  e3uw_3d, 'U', 1._wp, e3vw_3d, 'V', 1._wp,                       &   
                 &                     kfillmode = jpfillcopy )   ! do not put 0 over closed boundaries
         ENDIF
         !
         ztopbot(:,:,1) = REAL(k_top, wp)
         ztopbot(:,:,2) = REAL(k_bot, wp)
         CALL lbc_lnk( 'dom_zgr', ztopbot, 'T', 1._wp, kfillmode = jpfillcopy )   ! do not put 0 over closed boundaries
         k_top(:,:) = NINT(ztopbot(:,:,1))
         k_bot(:,:) = NINT(ztopbot(:,:,2))
         !
      ENDIF
      !
      ! the following is mandatory
      ! make sure that closed boundaries are correctly defined in k_top that will be used to compute all mask arrays
      !
      zmsk(:,:) = 1._wp                                       ! default: no closed boundaries
      IF( .NOT. l_Iperio ) THEN                                    ! E-W closed:
         zmsk(  mi0(     1+nn_hls,nn_hls):mi1(     1+nn_hls,nn_hls),:) = 0._wp   ! first column of inner global domain at 0
         zmsk(  mi0(jpiglo-nn_hls,nn_hls):mi1(jpiglo-nn_hls,nn_hls),:) = 0._wp   ! last  column of inner global domain at 0 
      ENDIF
      IF( .NOT. l_Jperio ) THEN                                    ! S closed:
         zmsk(:,mj0(     1+nn_hls,nn_hls):mj1(     1+nn_hls,nn_hls)  ) = 0._wp   ! first   line of inner global domain at 0
      ENDIF
      IF( .NOT. ( l_Jperio .OR. l_NFold ) ) THEN                   ! N closed:
         zmsk(:,mj0(jpjglo-nn_hls,nn_hls):mj1(jpjglo-nn_hls,nn_hls)  ) = 0._wp   ! last    line of inner global domain at 0
      ENDIF
      CALL lbc_lnk( 'usrdef_zgr', zmsk, 'T', 1._wp )             ! set halos
      k_top(:,:) = k_top(:,:) * NINT( zmsk(:,:) )
      !
      ! Any closed seas (defined by closea_mask > 0 in domain_cfg file) to be filled 
      ! in at runtime if ln_closea=.false.
      IF( ln_closea ) THEN
         IF ( ln_maskcs ) THEN
            ! mask all the closed sea
            CALL clo_msk( k_top, k_bot, mask_opnsea, 'mask_opensea' )
         ELSE IF ( ln_mask_csundef ) THEN
            ! defined closed sea are kept
            ! mask all the undefined closed sea
            CALL clo_msk( k_top, k_bot, mask_csundef, 'mask_csundef' )
         END IF
      END IF
      !
      IF(lwp) THEN                     ! Control print
         WRITE(numout,*)
         WRITE(numout,*) '   Type of vertical coordinate (read in ', TRIM( cn_domcfg ), ' file or set in userdef_zgr) :'
         WRITE(numout,*) '      z-coordinate - full steps      l_zco    = ', l_zco
         WRITE(numout,*) '      z-coordinate - partial steps   l_zps    = ', l_zps
         WRITE(numout,*) '      s- or hybrid z-s-coordinate    l_sco    = ', l_sco
         WRITE(numout,*) '      ice shelf cavities             ln_isfcav = ', ln_isfcav
      ENDIF

      ioptio = 0                       ! Check Vertical coordinate options
      IF( l_zco      )   ioptio = ioptio + 1
      IF( l_zps      )   ioptio = ioptio + 1
      IF( l_sco      )   ioptio = ioptio + 1
      IF( ioptio /= 1 )   CALL ctl_stop( ' none or several vertical coordinate options used' )


      !                                ! top/bottom ocean level indices for t-, u- and v-points (f-point also for top)
      CALL zgr_top_bot( k_top, k_bot, is_mbkuvf )      ! with a minimum value set to 1
      !
      IF( lk_vco_3d ) THEN
         !                                ! ice shelf draft and bathymetry
         DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )
            ikt = mikt(ji,jj)
            ikb = mbkt(ji,jj)
            bathy  (ji,jj) = gdepw_3d(ji,jj,ikb+1)
            risfdep(ji,jj) = gdepw_3d(ji,jj,ikt  )
         END_2D
      ELSE
         !                                ! ice shelf draft and bathymetry
         DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )
            ikt = mikt(ji,jj)
            ikb = mbkt(ji,jj)
            bathy  (ji,jj) = gdepw_1d(ikb+1)
            risfdep(ji,jj) = gdepw_1d(ikt  )
         END_2D
      ENDIF
      !
      !                                ! deepest/shallowest W level Above/Below ~10m
!!gm BUG in s-coordinate this does not work!
      zrefdep = 10._wp - 0.1_wp * MINVAL( e3w_1d )                   ! ref. depth with tolerance (10% of minimum layer thickness)
      nlb10 = MINLOC( gdepw_1d, mask = gdepw_1d > zrefdep, dim = 1 ) ! shallowest W level Below ~10m
      nla10 = nlb10 - 1                                              ! deepest    W level Above ~10m
!!gm end bug
      !
      IF( lwp )   THEN
         WRITE(numout,*) ' MIN val k_top   ', MINVAL(   k_top(:,:) ), ' MAX ', MAXVAL( k_top(:,:) )
         WRITE(numout,*) ' MIN val k_bot   ', MINVAL(   k_bot(:,:) ), ' MAX ', MAXVAL( k_bot(:,:) )
         IF( lk_vco_1d3d ) THEN
            WRITE(numout,*) ' MIN val e3    t ', MINVAL(   e3t_3d(:,:,:) ), ' f ', MINVAL(   e3f_3d(:,:,:) ),  &
               &                          ' u ', MINVAL(   e3u_3d(:,:,:) ), ' u ', MINVAL(   e3v_3d(:,:,:) )
            WRITE(numout,*) ' MAX val e3    t ', MAXVAL(   e3t_3d(:,:,:) ), ' f ', MAXVAL(   e3f_3d(:,:,:) ),  &
               &                          ' u ', MAXVAL(   e3u_3d(:,:,:) ), ' u ', MAXVAL(   e3v_3d(:,:,:) )
         ELSEIF( lk_vco_3d ) THEN
            WRITE(numout,*) ' MIN val depth t ', MINVAL( gdept_3d(:,:,:) ),   &
               &                          ' w ', MINVAL( gdepw_3d(:,:,:) )
            WRITE(numout,*) ' MIN val e3    t ', MINVAL(   e3t_3d(:,:,:) ), ' f ', MINVAL(   e3f_3d(:,:,:) ),  &
               &                          ' u ', MINVAL(   e3u_3d(:,:,:) ), ' u ', MINVAL(   e3v_3d(:,:,:) ),  &
               &                          ' uw', MINVAL(  e3uw_3d(:,:,:) ), ' vw', MINVAL(  e3vw_3d(:,:,:)),   &
               &                          ' w ', MINVAL(   e3w_3d(:,:,:) )
            WRITE(numout,*) ' MAX val depth t ', MAXVAL( gdept_3d(:,:,:) ),   &
               &                          ' w ', MINVAL( gdepw_3d(:,:,:) )
            WRITE(numout,*) ' MAX val e3    t ', MAXVAL(   e3t_3d(:,:,:) ), ' f ', MAXVAL(   e3f_3d(:,:,:) ),  &
               &                          ' u ', MAXVAL(   e3u_3d(:,:,:) ), ' u ', MAXVAL(   e3v_3d(:,:,:) ),  &
               &                          ' uw', MAXVAL(  e3uw_3d(:,:,:) ), ' vw', MAXVAL(  e3vw_3d(:,:,:) ),  &
               &                          ' w ', MAXVAL(   e3w_3d(:,:,:) )
         ENDIF
      ENDIF
      !
   END SUBROUTINE dom_zgr


!!st   SUBROUTINE zgr_read( ld_zco  , ld_zps  , ld_sco  , ld_isfcav,   &   ! type of vertical coordinate
!!st      &                 pdept_1d, pdepw_1d, pe3t_1d , pe3w_1d  ,   &   ! 1D reference vertical coordinate
!!st      &                 pdept , pdepw ,                            &   ! 3D t & w-points depth
!!st      &                 pe3t  , pe3u  , pe3v   , pe3f ,            &   ! vertical scale factors
!!st      &                 pe3w  , pe3uw , pe3vw         ,            &   !     -      -      -
!!st      &                 k_top  , k_bot  ,                          &   ! top & bottom ocean level
!!st      &                 k_mbkuvf  , k_bot_u  , k_bot_v  , k_bot_f  )   ! U/V/F points bottom levels
!!st      !!---------------------------------------------------------------------
!!st      !!              ***  ROUTINE zgr_read  ***
!!st      !!
!!st      !! ** Purpose :   Read the vertical information in the domain configuration file
!!st      !!
!!st      !!----------------------------------------------------------------------
!!st      LOGICAL                   , INTENT(out) ::   ld_zco, ld_zps, ld_sco      ! vertical coordinate flags
!!st      LOGICAL                   , INTENT(out) ::   ld_isfcav                   ! under iceshelf cavity flag
!!st      REAL(wp), DIMENSION(:)    , INTENT(out) ::   pdept_1d, pdepw_1d          ! 1D grid-point depth       [m]
!!st      REAL(wp), DIMENSION(:)    , INTENT(out) ::   pe3t_1d , pe3w_1d           ! 1D vertical scale factors [m]
!!st      REAL(wp), DIMENSION(:,:,:), INTENT(out) ::   pdept, pdepw                ! grid-point depth          [m]
!!st      REAL(wp), DIMENSION(:,:,:), INTENT(out) ::   pe3t , pe3u , pe3v , pe3f   ! vertical scale factors    [m]
!!st      REAL(wp), DIMENSION(:,:,:), INTENT(out) ::   pe3w , pe3uw, pe3vw         !    -       -      -
!!st      INTEGER , DIMENSION(:,:)  , INTENT(out) ::   k_top , k_bot               ! first & last ocean level
!!st      INTEGER                   , INTENT(out) ::   k_mbkuvf                    ! ==1 if mbku, mbkv, mbkf are in file
!!st      INTEGER , DIMENSION(:,:)  , INTENT(out) ::   k_bot_u , k_bot_v, k_bot_f  ! bottom levels at U/V/F points
!!st      !
!!st      INTEGER  ::   ji,jj,jk     ! dummy loop index
!!st      INTEGER  ::   inum, iatt
!!st      REAL(WP) ::   z_zco, z_zps, z_sco, z_cav
!!st      REAL(wp), DIMENSION(jpi,jpj) ::   z2d   ! 2D workspace
!!st      CHARACTER(len=7) ::   catt   ! 'zco', 'zps, 'sco' or 'UNKNOWN'
!!st      !!----------------------------------------------------------------------
!!st      !
!!st      IF(lwp) THEN
!!st         WRITE(numout,*)
!!st         WRITE(numout,*) '   zgr_read : read the vertical coordinates in ', TRIM( cn_domcfg ), ' file'
!!st         WRITE(numout,*) '   ~~~~~~~~'
!!st      ENDIF
!!st      !
!!st      CALL iom_open( cn_domcfg, inum )
!!st      !
!!st      !                          !* type of vertical coordinate
!!st      CALL iom_getatt( inum, 'VertCoord', catt )   ! returns 'UNKNOWN' if not found
!!st      ld_zco = catt == 'zco'          ! default = .false.
!!st      ld_zps = catt == 'zps'          ! default = .false.
!!st      ld_sco = catt == 'sco'          ! default = .false.
!!st      !                          !* ocean cavities under iceshelves
!!st      CALL iom_getatt( inum,    'IsfCav', iatt )   ! returns -999 if not found
!!st      ld_isfcav = iatt == 1           ! default = .false.
!!st      !
!!st      ! ------- keep compatibility with OLD VERSION... start -------
!!st      IF( catt == 'UNKNOWN' ) THEN
!!st         CALL iom_get( inum,    'ln_zco', z_zco )   ;   ld_zco = z_zco /= 0._wp
!!st         CALL iom_get( inum,    'ln_zps', z_zps )   ;   ld_zps = z_zps /= 0._wp
!!st         CALL iom_get( inum,    'ln_sco', z_sco )   ;   ld_sco = z_sco /= 0._wp
!!st      ENDIF
!!st      IF( iatt == -999 ) THEN
!!st         CALL iom_get( inum, 'ln_isfcav', z_cav )   ;   ld_isfcav = z_cav /= 0._wp
!!st      ENDIF
!!st      ! ------- keep compatibility with OLD VERSION... end -------
!!st      !
!!st      !                          !* ocean top and bottom level
!!st      CALL iom_get( inum, jpdom_global, 'top_level'    , z2d   )   ! 1st wet T-points (ISF)
!!st      k_top(:,:) = NINT( z2d(:,:) )
!!st      CALL iom_get( inum, jpdom_global, 'bottom_level' , z2d   )   ! last wet T-points
!!st      k_bot(:,:) = NINT( z2d(:,:) )
!!st      !
!!st      !                          !* vertical scale factors
!!st      CALL iom_get( inum, jpdom_unknown, 'e3t_1d'  , pe3t_1d  )                     ! 1D reference coordinate
!!st      CALL iom_get( inum, jpdom_unknown, 'e3w_1d'  , pe3w_1d  )
!!st      !
!!st      CALL iom_get( inum, jpdom_global, 'e3t_0'  , pe3t , cd_type = 'T', psgn = 1._wp, kfill = jpfillcopy )    ! 3D coordinate
!!st      CALL iom_get( inum, jpdom_global, 'e3u_0'  , pe3u , cd_type = 'U', psgn = 1._wp, kfill = jpfillcopy )
!!st      CALL iom_get( inum, jpdom_global, 'e3v_0'  , pe3v , cd_type = 'V', psgn = 1._wp, kfill = jpfillcopy )
!!st      CALL iom_get( inum, jpdom_global, 'e3f_0'  , pe3f , cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )
!!st      CALL iom_get( inum, jpdom_global, 'e3w_0'  , pe3w , cd_type = 'W', psgn = 1._wp, kfill = jpfillcopy )
!!st      CALL iom_get( inum, jpdom_global, 'e3uw_0' , pe3uw, cd_type = 'U', psgn = 1._wp, kfill = jpfillcopy )
!!st      CALL iom_get( inum, jpdom_global, 'e3vw_0' , pe3vw, cd_type = 'V', psgn = 1._wp, kfill = jpfillcopy )
!!st      !
!!st      !                          !* depths
!!st      !                                   !- old depth definition (obsolescent feature)
!!st      IF(  iom_varid( inum, 'gdept_1d', ldstop = .FALSE. ) > 0  .AND.  &
!!st         & iom_varid( inum, 'gdepw_1d', ldstop = .FALSE. ) > 0  .AND.  &
!!st         & iom_varid( inum, 'gdept_0' , ldstop = .FALSE. ) > 0  .AND.  &
!!st         & iom_varid( inum, 'gdepw_0' , ldstop = .FALSE. ) > 0    ) THEN
!!st         CALL ctl_warn( 'zgr_read : old definition of depths and scale factors used ', & 
!!st            &           '           depths at t- and w-points read in the domain configuration file')
!!st         CALL iom_get( inum, jpdom_unknown, 'gdept_1d', pdept_1d )   
!!st         CALL iom_get( inum, jpdom_unknown, 'gdepw_1d', pdepw_1d )
!!st         CALL iom_get( inum, jpdom_global , 'gdept_0' , pdept, kfill = jpfillcopy )
!!st         CALL iom_get( inum, jpdom_global , 'gdepw_0' , pdepw, kfill = jpfillcopy )
!!st         !
!!st      ELSE                                !- depths computed from e3. scale factors
!!st         CALL e3_to_depth( pe3t_1d, pe3w_1d, pdept_1d, pdepw_1d )    ! 1D reference depth
!!st         CALL e3_to_depth( pe3t   , pe3w   , pdept   , pdepw    )    ! 3D depths
!!st#if defined key_qco && key_isf
!!st         DO_3D( nn_hls, nn_hls, nn_hls, nn_hls, 2, jpk )        ! vertical sum at partial cell xxxx other level  
!!st            IF( jk == k_top(ji,jj) ) THEN                               ! first ocean point : partial cell
!!st               pdept(ji,jj,jk) = pdepw(ji,jj,jk  ) + 0.5_wp * pe3w(ji,jj,jk)   ! = risfdep + 1/2 e3w_0(mikt)
!!st            ELSE                                                        !  other levels
!!st               pdept(ji,jj,jk) = pdept(ji,jj,jk-1) +          pe3w(ji,jj,jk) 
!!st            ENDIF
!!st         END_3D
!!st#endif
!!st         IF(lwp) THEN
!!st            WRITE(numout,*)
!!st            WRITE(numout,*) '              Reference 1D z-coordinate depth and scale factors:'
!!st            WRITE(numout, "(9x,' level  gdept_1d  gdepw_1d  e3t_1d   e3w_1d  ')" )
!!st            WRITE(numout, "(10x, i4, 4f9.2)" ) ( jk, pdept_1d(jk), pdepw_1d(jk), pe3t_1d(jk), pe3w_1d(jk), jk = 1, jpk )
!!st         ENDIF
!!st      ENDIF
!!st      !
!!st      IF( iom_varid( inum, 'mbku', ldstop = .FALSE. ) > 0 ) THEN
!!st         IF(lwp) WRITE(numout,*) '          mbku, mbkv & mbkf read in ', TRIM(cn_domcfg), ' file'
!!st         CALL iom_get( inum, jpdom_global, 'mbku', z2d, cd_type = 'U', psgn = 1._wp, kfill = jpfillcopy )
!!st         k_bot_u(:,:) = NINT( z2d(:,:) )
!!st         CALL iom_get( inum, jpdom_global, 'mbkv', z2d, cd_type = 'V', psgn = 1._wp, kfill = jpfillcopy )
!!st         k_bot_v(:,:) = NINT( z2d(:,:) )
!!st         CALL iom_get( inum, jpdom_global, 'mbkf', z2d, cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )
!!st         k_bot_f(:,:) = NINT( z2d(:,:) )
!!st         k_mbkuvf = 1
!!st      ELSE
!!st         k_mbkuvf = 0
!!st      ENDIF
!!st      !
!!st      ! reference depth for negative bathy (wetting and drying only)
!!st      IF( ll_wd )  CALL iom_get( inum,  'rn_wd_ref_depth' , ssh_ref   )
!!st      !
!!st      CALL iom_close( inum )
!!st      !
!!st   END SUBROUTINE zgr_read


   SUBROUTINE zgr_top_bot( k_top, k_bot, k_mbkuvf )
      !!----------------------------------------------------------------------
      !!                    ***  ROUTINE zgr_top_bot  ***
      !!
      !! ** Purpose :   defines the vertical index of ocean bottom (mbk. arrays)
      !!
      !! ** Method  :   computes from k_top and k_bot with a minimum value of 1 over land
      !!
      !! ** Action  :   mikt, miku, mikv :   vertical indices of the shallowest 
      !!                                     ocean level at t-, u- & v-points
      !!                                     (min value = 1)
      !! ** Action  :   mbkt, mbku, mbkv :   vertical indices of the deeptest 
      !!                mbkf                 ocean level at t-, u-, v- & f-points
      !!                                     (min value = 1 over land)
      !!----------------------------------------------------------------------
      INTEGER , DIMENSION(:,:), INTENT(in) ::   k_top, k_bot   ! top & bottom ocean level indices
      INTEGER                 , INTENT(in) ::   k_mbkuvf       ! flag to recompute mbku, mbkv, mbkf
      !
      INTEGER ::   ji, jj   ! dummy loop indices
      REAL(wp), DIMENSION(jpi,jpj) ::   zk   ! workspace
      !!----------------------------------------------------------------------
      !
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) '    zgr_top_bot : ocean top and bottom k-index of T-, U-, V- and W-levels '
      IF(lwp) WRITE(numout,*) '    ~~~~~~~~~~~'
      !
      mikt(:,:) = MAX( k_top(:,:) , 1 )    ! top    ocean k-index of T-level (=1 over land)
      !
      mbkt(:,:) = MAX( k_bot(:,:) , 1 )    ! bottom ocean k-index of T-level (=1 over land)
 
      !                                    ! N.B.  top     k-index of W-level = mikt
      !                                    !       bottom  k-index of W-level = mbkt+1
      DO_2D( 0, 0, 0, 0 )
         miku(ji,jj) = MAX(  mikt(ji+1,jj  ) , mikt(ji,jj)  )
         mikv(ji,jj) = MAX(  mikt(ji  ,jj+1) , mikt(ji,jj)  )
         mikf(ji,jj) = MAX(  mikt(ji  ,jj+1) , mikt(ji,jj), mikt(ji+1,jj  ), mikt(ji+1,jj+1)  )
      END_2D

      IF ( k_mbkuvf==0 ) THEN
         IF(lwp) WRITE(numout,*) '         mbku, mbkv, mbkf computed from mbkt'
         DO_2D( 0, 0, 0, 0 )
            mbku(ji,jj) = MIN(  mbkt(ji+1,jj  ) , mbkt(ji,jj)  )
            mbkv(ji,jj) = MIN(  mbkt(ji  ,jj+1) , mbkt(ji,jj)  )
            mbkf(ji,jj) = MIN(  mbkt(ji  ,jj+1) , mbkt(ji,jj), mbkt(ji+1,jj  ), mbkt(ji+1,jj+1)  )
         END_2D
      ELSE
         IF(lwp) WRITE(numout,*) '         mbku, mbkv, mbkf read from file'
         ! Use mbku, mbkv, mbkf from file
         ! Ensure these are lower than expected bottom level deduced from mbkt
         DO_2D( 0, 0, 0, 0 )
            mbku(ji,jj) = MIN(  mbku(ji,jj), mbkt(ji+1,jj  ) , mbkt(ji,jj)  )
            mbkv(ji,jj) = MIN(  mbkv(ji,jj), mbkt(ji  ,jj+1) , mbkt(ji,jj)  )
            mbkf(ji,jj) = MIN(  mbkf(ji,jj), mbkt(ji  ,jj+1) , mbkt(ji,jj), mbkt(ji+1,jj  ), mbkt(ji+1,jj+1)  )
         END_2D
      ENDIF
      ! convert into REAL to use lbc_lnk ; impose a min value of 1 as a zero can be set in lbclnk 
      DO_2D( 0, 0, 0, 0 )
         zk(ji,jj) = REAL( miku(ji,jj), wp )
      END_2D
      CALL lbc_lnk( 'domzgr', zk, 'U', 1.0_wp )
      miku(:,:) = MAX( NINT( zk(:,:) ), 1 )

      DO_2D( 0, 0, 0, 0 )
         zk(ji,jj) = REAL( mikv(ji,jj), wp )
      END_2D
      CALL lbc_lnk( 'domzgr', zk, 'V', 1.0_wp )
      mikv(:,:) = MAX( NINT( zk(:,:) ), 1 )
      
      DO_2D( 0, 0, 0, 0 )
         zk(ji,jj) = REAL( mikf(ji,jj), wp )
      END_2D
      CALL lbc_lnk( 'domzgr', zk, 'F', 1.0_wp )
      mikf(:,:) = MAX( NINT( zk(:,:) ), 1 )
      !
      DO_2D( 0, 0, 0, 0 )
         zk(ji,jj) = REAL( mbku(ji,jj), wp )
      END_2D
      CALL lbc_lnk( 'domzgr', zk, 'U', 1.0_wp )
      mbku(:,:) = MAX( NINT( zk(:,:) ), 1 )
      
      DO_2D( 0, 0, 0, 0 )
         zk(ji,jj) = REAL( mbkv(ji,jj), wp )
      END_2D
      CALL lbc_lnk( 'domzgr', zk, 'V', 1.0_wp )
      mbkv(:,:) = MAX( NINT( zk(:,:) ), 1 )

      DO_2D( 0, 0, 0, 0 )
         zk(ji,jj) = REAL( mbkf(ji,jj), wp )
      END_2D
      CALL lbc_lnk( 'domzgr', zk, 'F', 1.0_wp )
      mbkf(:,:) = MAX( NINT( zk(:,:) ), 1 )
      !
   END SUBROUTINE zgr_top_bot

   !!======================================================================
END MODULE domzgr
