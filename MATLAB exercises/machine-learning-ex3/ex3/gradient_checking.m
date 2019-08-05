function gradApprox=gradChecking(X,y,theta,EPSILON)
  n=size(theta,1);
  gradApprox=zeros(n,1);
  for i=1:n
    thetaPlus=theta;
    thetaMinus=theta;
    thetaPlus(i)=thetaPlus(i)+EPSILON;
    thetaMinus(i)=thetaMinus(i)-EPSILON;
    gradApprox(i)=(costFunction(X,y,thetaPlus)-costFunction(X,y,thetaMinus))/(2*EPSILON);
  endfor
  % Now I should check if gradApprox == DVec (that is the result got from 
  %                   backpropagation, at least for some cycles)
end