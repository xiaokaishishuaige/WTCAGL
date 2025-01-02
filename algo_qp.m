function [obj,results_log] = algo_qp(X, X_bar, Y1, k,lambda,beta,a,omega,gamma,d)

%% initialize
v = length(X);
n = size(Y1, 1);

for i = 1:v
   dd(i) = size(X{i}, 1);
end
d_bar = size(X_bar, 1);

mu = 0.01; max_mu = 10e10; pho_mu = 2;
%%%初始化
A = cell(1,v); 
Z = cell(1,v);
Y = cell(1,v);
C = cell(1,v);
%d=k;
 m=a*k;
for i = 1:v
    Z{i} = zeros(m,n);
    Y{i} = zeros(m,n);
    C{i} = zeros(m,n);
    A{i} = zeros(d,m);
    W{i}=zeros(dd(i),d);
end
for i = 1:v
    Lb{i} = zeros(m,m);
     B{i} = zeros(m,m);
end
Lb{v+1} = zeros(m,m);
B{v+1} = zeros(m,m);
D = zeros(d_bar, m);
Z{v+1} = zeros(m, n);
Y{v+1}=zeros(m,n);
C{v+1}=zeros(m,n);
y = zeros(m*n*v,1);
c = zeros(m*n*v,1);
sX = [m, n, v+1];
iter = 1;
while(iter <100)
    
    for i=1:v
        R{i}= X{i}*Z{i}'*A{i}';    
        [U,~,G] = svd(R{i},'econ');
        W{i} = U*G';
    end    
   clear U G    
%更新A
   for i=1:v
        RR{i}= W{i}'*X{i}*Z{i}';      
        [U,~,G] = svd(RR{i},'econ');
        A{i} = U*G';
    end    
   clear U G 
[U,~,V]=svd(lambda*X_bar*Z{v+1}','econ');
D=U*V';
clear U V
 for i = 1 :v
    B{i} = constructW_PKN(A{i}, 3);
    Db = diag(sum(B{i},1)+eps);
    Lb{i} = eye(m,m)-Db^-0.5*B{i}*Db^-0.5;
 end 
  for i = 1:v
Z{i}=(2*A{i}'*A{i}+mu*eye(m)+2*Lb{i}*gamma)\(2*A{i}'*W{i}'*X{i}-Y{i}+mu*C{i});
  end
  B{v+1} = constructW_PKN(D, 3);
    Db = diag(sum(B{v+1},1)+eps);
    Lb{v+1} = eye(m,m)-Db^-0.5*B{v+1}*Db^-0.5;
%更新Z_bar
Z{v+1}=(2*(D'*D)*lambda+mu*eye(m)+2*Lb{v+1}*gamma)\(2*lambda*D'*X_bar-Y{v+1}+mu*C{v+1});
%更新C
Z_tensor=cat(3,Z{:,:});
Y_tensor=cat(3,Y{:,:});
z=Z_tensor(:);
y=Y_tensor(:);
[c, ~]=wshrinkObj(z+1/mu*y,beta/mu,sX,0,3,omega);
C_tensor=reshape(c,sX);

for inter_C=1:v+1
 C{inter_C}=C_tensor(:,:,inter_C);
end
y=y+mu*(z-c);
Y_tensor=reshape(y,sX);
for inter_Y = 1:v+1
        Y{inter_Y} = Y_tensor(:,:,inter_Y);
end

mu = min(mu*pho_mu, max_mu);


for iv=1:v
  leq{iv}=Z{iv}-C{iv};
end
leqm = cat(3,leq{:,:});
leqm2 = max(abs(leqm(:)));
err = max(leqm2);
obj(iter)=err;

iter=iter+1;
end
Sbar=[];
for i = 1:v+1
    Sbar=cat(1,Sbar,1/sqrt(v+1)*Z{i});
end

[Q,~,~] = mySVD(Sbar',k); 

 rng(4234,'twister') % set random seed for re-production
labels=litekmeans(Q, k, 'MaxIter', 100,'Replicates',10);

NMI = nmi(labels,Y1);
Purity = purity(Y1, labels);
ACC = Accuracy(labels,double(Y1));
[Fscore,Precision,~] = compute_f(Y1,labels);
[AR,RI,~,~]=RandIndex(Y1,labels);
results_log = [NMI,ACC,Purity,Fscore,Precision,AR,RI];





