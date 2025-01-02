clear;
clc;

data_name = 'NGs';
fprintf('\ndata_name: %s', data_name);

load([ data_name, '.mat']);

 Y1=truelabel{1}';
 X=data;
k = length(unique(Y1));
V = length(X);
 
for v = 1 : V   
    X{v} =X{v}';
end
X_bar = [];
for v = 1 : V   
    X_bar = [X_bar,X{v}];
    X{v} = mapstd(X{v},0,1)';
end

X_bar = mapstd(X_bar,0,1)';
d=k;
omega = [1,1,1,1];
   param1 = [1];
  param3=[3];
  %lambda=1;
  beta = 1;
  gamma=1;

 for i = param3
      a=i;
  for ii=param1
  lambda = ii;
  fprintf('params:\t a=%f lambda=%f \t beta=%f gamma=%f \n',a, lambda,beta,gamma);
          tic;
          [obj,results_log] = algo_qp(X, X_bar, Y1, k,lambda,beta,a,omega,gamma,d);
          tt=toc;
          fprintf('result:\tNMI:%f, ACC:%f, Purity:%f, Fsocre:%f, Precision:%f, AR:%f,RI:%f,times:%f\n',results_log,tt);
   end
 end