! MOM6 function below is from the Modular Ocean Model version 6.
! SPDX-License-Identifier: Apache-2.0
! The unused receiver type and driver are only a component harness.
program reference
use gsw_mod_toolbox, only: gsw_rho, gsw_rho_first_derivatives, gsw_sr_from_sp
use gsw_mod_toolbox, only: gsw_ct_from_pt,gsw_pt_from_ct,gsw_ct_from_t, &
 gsw_pt0_from_t,gsw_t_from_ct,gsw_ct_first_derivatives,gsw_rho_t_exact,gsw_ct_freezing_poly
implicit none
type TEOS10_EOS
end type
type(TEOS10_EOS) :: eos
real, parameter :: Pa2db  = 1.e-4  !< The conversion factor from Pa to dbar [dbar Pa-1]
real :: s,t,p,rs,rt,rp,cts,ctt
integer :: ios
do
 read(*,*,iostat=ios) s,t,p
 if(ios /= 0) exit
 call gsw_rho_first_derivatives(s,t,p,rs,rt,rp)
 call gsw_ct_first_derivatives(s,t,cts,ctt)
 write(*,'(15(es26.18,1x))') gsw_rho(s,t,p),rs,rt,rp,gsw_sr_from_sp(s), &
   density_elem_TEOS10(eos,t,s,p*1.e4), &
   gsw_ct_from_pt(s,t),gsw_pt_from_ct(s,t),gsw_ct_from_t(s,t,p),gsw_pt0_from_t(s,t,p), &
   gsw_t_from_ct(s,t,p),cts,ctt,gsw_rho_t_exact(s,gsw_t_from_ct(s,t,p),p), &
   gsw_ct_freezing_poly(s,0.,0.)
enddo
contains
real elemental function density_elem_TEOS10(this, T, S, pressure)
  class(TEOS10_EOS), intent(in) :: this     !< This EOS
  real,              intent(in) :: T        !< Conservative temperature [degC].
  real,              intent(in) :: S        !< Absolute salinity [g kg-1].
  real,              intent(in) :: pressure !< pressure [Pa].

  density_elem_TEOS10 = gsw_rho(S, T, pressure * Pa2db)

end function density_elem_TEOS10
end program
