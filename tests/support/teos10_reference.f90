! MOM6 function below is from the Modular Ocean Model version 6.
! SPDX-License-Identifier: Apache-2.0
! The unused receiver type and driver are only a component harness.
program reference
use gsw_mod_toolbox, only: gsw_rho, gsw_rho_first_derivatives, gsw_sr_from_sp
implicit none
type TEOS10_EOS
end type
type(TEOS10_EOS) :: eos
real, parameter :: Pa2db=1.e-4
real :: s,t,p,rs,rt,rp
integer :: ios
do
 read(*,*,iostat=ios) s,t,p
 if(ios /= 0) exit
 call gsw_rho_first_derivatives(s,t,p,rs,rt,rp)
 write(*,'(6(es26.18,1x))') gsw_rho(s,t,p),rs,rt,rp,gsw_sr_from_sp(s), &
   density_elem_TEOS10(eos,t,s,p*1.e4)
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
