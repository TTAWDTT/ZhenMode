! Independent reference driver. Compile with the unmodified author routine:
! HiroyukiTsujino/JRA55-do @ 30c8e1a84386c1db8a436d980826b884b2075089
! anl/diagflux/src/bulk-ncar.F90; no OGCM_LYCOEF or CALHEIGHT macro.
! The explicit project scalar floor is applied before calling the reference.
program reference
  implicit none
  integer, parameter :: n=8
  integer :: i
  real(8) :: ta(n,1),q(n,1),u(n,1),v(n,1),w(n,1),p(n,1),sst(n,1),mask(n,1)
  real(8) :: tx(n,1),ty(n,1),lh(n,1),sh(n,1),e(n,1)
  real(8) :: tu(n,1),qu(n,1),dt(n,1),dq(n,1),un(n,1),rho,cp,cd,ch,ce
  ta(:,1)=[30d0,10d0,20d0,18d0,22d0,15d0,-5d0,27d0]
  sst(:,1)=[25d0,18d0,21d0,20d0,20d0,10d0,-1d0,25d0]
  q(:,1)=[.012d0,.004d0,.013d0,.004d0,.006d0,.004d0,.001d0,.015d0]
  u(:,1)=[5d0,8d0,.2d0,35d0,50d0,1d0,10d0,20d0]
  v=0d0; p=1013.25d0; mask=1d0
  w=max(abs(u),.5d0)
  call bulk(tx,ty,lh,sh,e,tu,qu,dt,dq,un,u,v,ta,q,w,p,sst,n,1,mask,10d0,10d0,10d0)
  do i=1,n
    rho=p(i,1)*100d0/(287.04d0*(ta(i,1)+273.15d0)*(1d0+(28.966d0/18.016d0-1d0)*q(i,1)))
    cp=1004.6d0*(1d0+.8735d0*q(i,1))
    cd=tx(i,1)/(rho*w(i,1)*u(i,1))
    ch=-sh(i,1)/(rho*cp*w(i,1)*dt(i,1))
    ce=e(i,1)/(rho*w(i,1)*dq(i,1))
    write(*,'(9(ES25.16E3,1X))') ta(i,1)+273.15d0,q(i,1),sst(i,1)+273.15d0, &
      q(i,1)+dq(i,1),w(i,1),cd,ch,ce,lh(i,1)
  end do
end program
