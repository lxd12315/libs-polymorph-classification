function probe_coral_domain_alignment(projectRoot)
%PROBE_CORAL_DOMAIN_ALIGNMENT Diagnostic only; target labels never tune model.
if nargin<1,projectRoot=fileparts(fileparts(fileparts(mfilename('fullpath'))));end
S=load(fullfile(projectRoot,'data','classification','julei.mat'),'spec_al','spec_al2');
W=load(fullfile(projectRoot,'data','classification','w.mat'),'w');
X=S.spec_al(1:12286,1:37)';Xt=S.spec_al2(1:12286,1:37)';
[~,X]=overlap_fuse_spectra(W.w,X,0.06);[~,Xt]=overlap_fuse_spectra(W.w,Xt,0.06);
y=[ones(24,1);2*ones(13,1)];sp=readtable(fullfile(projectRoot,'reports','fixed_sample_split.csv'));im=logical(sp.Modeling_set);
X=snv(X);Xt=snv(Xt);mu=mean(X(im,:),1);sd=std(X(im,:),0,1);keep=sd>1e-10;sd=sd(keep);mu=mu(keep);
[coef,Zs]=pca((X(im,keep)-mu)./sd,'NumComponents',3);zm=mean(Zs,1);zs=std(Zs,0,1);Zs=(Zs-zm)./zs;
Zt=((((Xt(:,keep)-mu)./sd)*coef)-zm)./zs;
mdl=fitcsvm(Zs,y(im),'KernelFunction','rbf','BoxConstraint',1,'KernelScale',1,'Standardize',false,'ClassNames',[1 2]);
alphas=[0 .25 .5 .75 1];acc=zeros(size(alphas));bacc=acc;err=cell(size(alphas));
Cs=cov(Zs)+1e-4*eye(3);Ct=cov(Zt)+1e-4*eye(3);A=mpower(Ct,-0.5)*mpower(Cs,0.5);mt=mean(Zt,1);ms=mean(Zs,1);
for i=1:numel(alphas)
    a=alphas(i);Za=(Zt-mt)*((1-a)*eye(3)+a*A)+(1-a)*mt+a*ms;p=predict(mdl,Za);
    acc(i)=mean(p==y);bacc(i)=.5*(mean(p(y==1)==1)+mean(p(y==2)==2));err{i}=find(p~=y)';
end
T=table(alphas',acc',bacc','VariableNames',{'Alignment_strength','Accuracy','Balanced_accuracy'});
writetable(T,fullfile(projectRoot,'reports','coral_probe_metrics.csv'));save(fullfile(projectRoot,'reports','coral_probe.mat'),'T','err','Zs','Zt','A');
disp(T);for i=1:numel(alphas),fprintf('alpha %.2f errors ',alphas(i));disp(err{i});end
end
function X=snv(X),X=(X-mean(X,2))./max(std(X,0,2),eps);end
