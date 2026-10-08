! Call the pinned, already compiled SIS2 implementation; no copied physics.
program sis2_native_server
  use mpp_domains_mod, only: mpp_domains_set_stack_size
  use fms_mod, only: fms_init, fms_end
  use mpp_mod, only: mpp_init, mpp_pe, mpp_npes
  use diag_manager_mod, only: diag_manager_init, diag_manager_set_time_end
  use time_manager_mod, only: time_type, set_date, set_time, set_calendar_type, GREGORIAN, operator(+), operator(*)
  use MOM_domains, only: AGRID
  use ice_model_mod, only: ice_data_type, ocean_ice_boundary_type, atmos_ice_boundary_type, land_ice_boundary_type, &
       ice_model_init, ice_model_restart, unpack_ocean_ice_boundary, update_ice_model_fast, &
       ice_model_fast_cleanup, unpack_land_ice_boundary, exchange_fast_to_slow_ice, update_ice_model_slow, ice_stock_pe, exchange_slow_to_fast_ice, set_ice_surface_fields
  use SIS2_ice_thm, only: get_SIS2_thermo_coefs
  use surface_flux_mod, only: surface_flux, surface_flux_init
  use sat_vapor_pres_mod, only: sat_vapor_pres_init
  use stock_constants_mod, only: ISTOCK_WATER, ISTOCK_HEAT, ISTOCK_SALT
  implicit none
  type(ice_data_type) :: ice
  type(ocean_ice_boundary_type) :: ocean
  type(atmos_ice_boundary_type) :: air
  type(land_ice_boundary_type) :: land
  type(time_type) :: clock, origin, interval
  real :: mass, heat, salt, fusion, cp_water, rho_ice, energy_scale
  integer :: gi0, gi1, gj0, gj1
  integer :: nx, ny, nc, is, ie, js, je, k0, k1, unit, io, step_number, dt, max_steps
  character(len=40) :: argument
  integer :: stamp(6), elapsed, stack_words
  logical :: advance, bulk
  integer :: k
  real, allocatable :: weather(:,:,:), result(:,:,:), qs(:,:), z10(:,:), rough_scale(:,:), gust(:,:), bulk_result(:,:,:,:)
  character(len=1000) :: request
  if (storage_size(mass) /= 64) error stop "bridge requires native 64-bit real"
  call get_command_argument(1,argument)
  read(argument,*,iostat=io) dt
  if (io /= 0) error stop "positive integer ice timestep required"
  call get_command_argument(2,argument)
  read(argument,*,iostat=io) max_steps
  if (io /= 0) error stop "positive integer ice step limit required"
  if (dt <= 0 .or. max_steps <= 0) error stop "positive interval and step limit required"
  call get_command_argument(5,argument)
  read(argument,*,iostat=io) stack_words
  if (io /= 0 .or. stack_words <= 0) error stop "positive MPI domains stack required"
  call mpp_init()
  if (mpp_npes() /= 1) error stop "bridge currently supports exactly one MPI rank"
  call fms_init()
  call mpp_domains_set_stack_size(stack_words)
  call sat_vapor_pres_init()
  call surface_flux_init()
  call set_calendar_type(GREGORIAN)
  call get_command_argument(3,argument)
  read(argument,'(i4,1x,i2,1x,i2,1x,i2,1x,i2,1x,i2)',iostat=io) stamp
  if (io /= 0) error stop "Gregorian start date required"
  call get_command_argument(4,argument)
  read(argument,*,iostat=io) elapsed
  if (io /= 0 .or. elapsed < 0) error stop "nonnegative elapsed seconds required"
  origin=set_date(stamp(1),stamp(2),stamp(3),stamp(4),stamp(5),stamp(6))
  clock=origin+set_time(mod(elapsed,86400),elapsed/86400)
  interval=set_time(mod(dt,86400),dt/86400)
  call diag_manager_init(TIME_INIT=stamp)
  ice%pe=.true.; ice%fast_ice_pe=.true.; ice%slow_ice_pe=.true.
  allocate(ice%pelist(1),ice%fast_pelist(1),ice%slow_pelist(1))
  ice%pelist=mpp_pe();ice%fast_pelist=mpp_pe();ice%slow_pelist=mpp_pe()
  call diag_manager_set_time_end(clock+interval*max_steps)
  call ice_model_init(ice,origin,clock,interval,interval)
  if (ice%flux_uv_stagger /= AGRID) error stop "bridge requires ICE_OCEAN_STRESS_STAGGER A"
  gi0=ice%sCS%G%isc;gi1=ice%sCS%G%iec;gj0=ice%sCS%G%jsc;gj1=ice%sCS%G%jec
  energy_scale=ice%sCS%US%Q_to_J_kg*ice%sCS%US%RZ_to_kg_m2
  call get_SIS2_thermo_coefs(ice%sCS%IST%ITV,Latent_fusion=fusion,Cp_water=cp_water,rho_ice=rho_ice)
  fusion=fusion*ice%sCS%US%Q_to_J_kg
  cp_water=cp_water*ice%sCS%US%Q_to_J_kg*ice%sCS%US%degC_to_C
  rho_ice=rho_ice*ice%sCS%US%R_to_kg_m3
  nx=size(ice%part_size,1);ny=size(ice%part_size,2);nc=size(ice%part_size,3)
  is=lbound(ice%part_size,1);ie=ubound(ice%part_size,1)
  js=lbound(ice%part_size,2);je=ubound(ice%part_size,2)
  k0=lbound(ice%part_size,3);k1=ubound(ice%part_size,3)

  allocate(ocean%u(is:ie,js:je),ocean%v(is:ie,js:je),ocean%t(is:ie,js:je),ocean%s(is:ie,js:je), &
           ocean%frazil(is:ie,js:je),ocean%sea_level(is:ie,js:je))
  allocate(air%u_flux(is:ie,js:je,k0:k1),air%v_flux(is:ie,js:je,k0:k1),air%u_star(is:ie,js:je,k0:k1), &
           air%t_flux(is:ie,js:je,k0:k1),air%q_flux(is:ie,js:je,k0:k1),air%lw_flux(is:ie,js:je,k0:k1), &
           air%sw_flux_vis_dir(is:ie,js:je,k0:k1),air%sw_flux_vis_dif(is:ie,js:je,k0:k1), &
           air%sw_flux_nir_dir(is:ie,js:je,k0:k1),air%sw_flux_nir_dif(is:ie,js:je,k0:k1), &
           air%sw_down_vis_dir(is:ie,js:je,k0:k1),air%sw_down_vis_dif(is:ie,js:je,k0:k1), &
           air%sw_down_nir_dir(is:ie,js:je,k0:k1),air%sw_down_nir_dif(is:ie,js:je,k0:k1), &
           air%lprec(is:ie,js:je,k0:k1),air%fprec(is:ie,js:je,k0:k1),air%dhdt(is:ie,js:je,k0:k1), &
           air%dedt(is:ie,js:je,k0:k1),air%drdt(is:ie,js:je,k0:k1),air%coszen(is:ie,js:je,k0:k1),air%p(is:ie,js:je,k0:k1))
  allocate(land%runoff(is:ie,js:je),land%calving(is:ie,js:je), &
           land%runoff_hflx(is:ie,js:je),land%calving_hflx(is:ie,js:je))
  allocate(weather(nx,ny,5),result(nx,ny,20),qs(nx,ny),z10(nx,ny),rough_scale(nx,ny),gust(nx,ny), &
           bulk_result(nx,ny,nc,13))
  z10=10.;rough_scale=1.e-4;gust=0.
  ocean%stagger=AGRID
  step_number=0
  write(*,*) 'ZMSIS_READY',1,nx,ny,nc,dt
  flush(6)
  do
    read(*,'(A)',iostat=io) request
    if (io /= 0 .or. trim(request)=='STOP') exit
    if (trim(request)=='SAVE') then
      ice%restart_output_dir='./RESTART/'
      call ice_model_restart(ice)
      write(*,*) 'ZMSIS_SAVED',step_number
      flush(6)
      cycle
    endif
    advance=request(1:5)=='STEP '
    bulk=request(1:5)=='BULK '
    if (.not.advance .and. .not.bulk .and. request(1:8)/='SURFACE ') error stop 'unknown bridge command'
    if (advance .and. step_number >= max_steps) error stop 'ice step limit exceeded'
    if (advance .or. bulk) then
      request=request(6:)
    else
      request=request(9:)
    endif
    open(newunit=unit,file=trim(request),access='stream',form='unformatted',status='old')
    read(unit) ocean%u,ocean%v,ocean%t,ocean%s,ocean%frazil,ocean%sea_level
    if (bulk) read(unit) weather
    if (advance) read(unit) air%u_flux,air%v_flux,air%u_star,air%t_flux,air%q_flux,air%lw_flux,air%sw_flux_vis_dir,air%sw_flux_vis_dif,air%sw_flux_nir_dir,air%sw_flux_nir_dif,air%sw_down_vis_dir,air%sw_down_vis_dif,air%sw_down_nir_dir,air%sw_down_nir_dif,air%lprec,air%fprec,air%dhdt,air%dedt,air%drdt,air%coszen,air%p
    if (advance) read(unit) land%runoff,land%calving,land%runoff_hflx,land%calving_hflx
    close(unit)
    call unpack_ocean_ice_boundary(ocean,ice)
    call exchange_slow_to_fast_ice(ice)
    call set_ice_surface_fields(ice)
    if (bulk) then
      do k=k0,k1
        qs=0.
        call surface_flux(weather(:,:,1),weather(:,:,2),weather(:,:,4),weather(:,:,5),weather(:,:,3), &
          z10,weather(:,:,3),ice%t_surf(:,:,k),weather(:,:,1),qs,ice%u_surf(:,:,k),ice%v_surf(:,:,k), &
          ice%rough_mom(:,:,k),ice%rough_heat(:,:,k),ice%rough_moist(:,:,k),rough_scale,gust, &
          result(:,:,1),result(:,:,2),result(:,:,3),result(:,:,4),result(:,:,5),result(:,:,6),result(:,:,7),result(:,:,8),result(:,:,9),result(:,:,10),result(:,:,11),result(:,:,12),result(:,:,13),result(:,:,14),result(:,:,15),result(:,:,16),result(:,:,17),result(:,:,18),result(:,:,19),result(:,:,20),real(dt),ice%area<=0.,spread(spread(k==k0,1,nx),2,ny),ice%area>0.)
        bulk_result(:,:,k-k0+1,1)=result(:,:,1)
        bulk_result(:,:,k-k0+1,2)=result(:,:,2)
        bulk_result(:,:,k-k0+1,3)=result(:,:,3)
        bulk_result(:,:,k-k0+1,4)=-result(:,:,4)
        bulk_result(:,:,k-k0+1,5)=-result(:,:,5)
        bulk_result(:,:,k-k0+1,6)=result(:,:,10)
        bulk_result(:,:,k-k0+1,7)=result(:,:,13)
        bulk_result(:,:,k-k0+1,8)=result(:,:,14)
        bulk_result(:,:,k-k0+1,9)=result(:,:,16)
        bulk_result(:,:,k-k0+1,10:12)=result(:,:,6:8)
        bulk_result(:,:,k-k0+1,13)=qs
      enddo
      open(newunit=unit,file=trim(request)//'.out',access='stream',form='unformatted',status='replace')
      write(unit) bulk_result
      close(unit)
      write(*,*) 'ZMSIS_BULK',step_number
      flush(6)
      cycle
    endif
  if (advance) then
  call update_ice_model_fast(air,ice)
  call ice_model_fast_cleanup(ice)
  call unpack_land_ice_boundary(ice,land)
  call exchange_fast_to_slow_ice(ice)
  call update_ice_model_slow(ice)
  call exchange_slow_to_fast_ice(ice)
  call set_ice_surface_fields(ice)
  endif
    call ice_stock_pe(ice,ISTOCK_WATER,mass)
    call ice_stock_pe(ice,ISTOCK_HEAT,heat)
    call ice_stock_pe(ice,ISTOCK_SALT,salt)
    open(newunit=unit,file=trim(request)//'.out',access='stream',form='unformatted',status='replace')
    if (advance) write(unit) ice%flux_u,ice%flux_v,ice%flux_t,ice%flux_q,ice%flux_lw,ice%flux_lh,ice%flux_sw_vis_dir,ice%flux_sw_vis_dif,ice%flux_sw_nir_dir,ice%flux_sw_nir_dif,ice%lprec,ice%fprec,ice%runoff,ice%calving,ice%runoff_hflx,ice%calving_hflx,ice%flux_salt,ice%mi,ice%p_surf
    if (advance) write(unit) &
      energy_scale*ice%sCS%IOF%Enth_Mass_in_ocn(gi0:gi1,gj0:gj1), &
      energy_scale*ice%sCS%IOF%Enth_Mass_out_ocn(gi0:gi1,gj0:gj1), &
      energy_scale*ice%sCS%IOF%Enth_Mass_in_atm(gi0:gi1,gj0:gj1), &
      energy_scale*ice%sCS%IOF%Enth_Mass_out_atm(gi0:gi1,gj0:gj1), &
      energy_scale*ice%sCS%FIA%frazil_left(gi0:gi1,gj0:gj1)
    write(unit) ice%part_size,ice%t_surf,ice%u_surf,ice%v_surf, &
      ice%rough_mom,ice%rough_heat,ice%rough_moist,ice%albedo_vis_dir,ice%albedo_vis_dif, &
      ice%albedo_nir_dir,ice%albedo_nir_dif,ice%s_surf,ice%area
    write(unit) mass,heat,salt,fusion,cp_water,rho_ice
    close(unit)
    if (advance) step_number=step_number+1
    write(*,*) 'ZMSIS_REPLY',step_number
    flush(6)
  enddo
  call fms_end()
end program
