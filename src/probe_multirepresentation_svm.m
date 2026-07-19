function probe_multirepresentation_svm(projectRoot)
%PROBE_MULTIREPRESENTATION_SVM SelF-Rocket-style representation selection probe.
if nargin<1,projectRoot=fileparts(fileparts(fileparts(mfilename('fullpath'))));end
rng(20260718,'twister');
S=load(fullfile(projectRoot,'data','classification','julei.mat'),'spec_al','spec_al2');W=load(fullfile(projectRoot,'data','classification','w.mat'),'w');
X=S.spec_al(1:12286,1:37)';Xt=S.spec_al2(1:12286,1:37)';[~,X]=overlap_fuse_spectra(W.w,X,.06);[~,Xt]=overlap_fuse_spectra(W.w,Xt,.06);
y=[ones(24,1);2*ones(13,1)];sp=readtable(fullfile(projectRoot,'reports','fixed_sample_split.csv'));im=logical(sp.Modeling_set);it=logical(sp.Final_test_set);
R=representations(X);Rt=representations(Xt);names={'Raw','D1','D2','Raw+D1','Raw+D1+D2'};
res=zeros(5,8);best=cell(5,1);predRob=cell(5,1);
for fam=1:5
 C=configs(fam);scores=zeros(numel(C),10);q=0;Xm=subsetReps(R,im);ym=y(im);
 for rep=1:2
  rng(20260718+rep);cv=cvpartition(ym,'KFold',5);
  for k=1:5,q=q+1;tr=training(cv,k);va=test(cv,k);
   for c=1:numel(C),m=fitM(Xm,ym,tr,C(c));p=predictM(m,Xm,va);scores(c,q)=bacc(ym(va),p);end
  end
 end
 mu=mean(scores,2);sd=std(scores,0,2);[~,ib]=max(mu-.05*sd);best{fam}=C(ib);m=fitM(R,y,im,C(ib));ph=predictM(m,R,it);pr=predictM(m,Rt,true(37,1));predRob{fam}=pr;
 res(fam,:)=[mu(ib),sd(ib),mean(ph==y(it)),bacc(y(it),ph),mean(pr==y),bacc(y,pr),macroF1(y,pr),sum(pr~=y)];
 fprintf('%s CV %.3f hold %.3f robust %.3f errors ',names{fam},res(fam,1),res(fam,3),res(fam,5));disp(find(pr~=y)');
end
T=array2table(res,'VariableNames',{'CV_BAcc','CV_SD','Holdout_Acc','Holdout_BAcc','Robust_Acc','Robust_BAcc','Robust_F1','Robust_errors'});T.Model=names';T=movevars(T,'Model','Before',1);
writetable(T,fullfile(projectRoot,'reports','multirepresentation_probe.csv'));save(fullfile(projectRoot,'reports','multirepresentation_probe.mat'),'T','best','predRob');disp(T);
end
function R=representations(X)
raw=snv(X);sm=smoothdata(raw,2,'sgolay',15);d1=snv(gradient(sm,1,2));d2=snv(gradient(d1,1,2));R={raw,d1,d2,raw,{raw,d1},{raw,d1,d2}};R=R([1 2 3 5 6]);
end
function C=configs(fam)
q=0;C=struct('fam',{},'pc',{},'a1',{},'a2',{},'box',{},'scale',{});
if fam<=3,aa=0;bb=0;else,aa=[.25 .5 1 2];if fam==5,bb=[.25 .5 1];else,bb=0;end,end
for pc=[2 3 5],for a=aa,for b=bb,for box=[.3 1 3],for sc=[1 3],q=q+1;C(q)=struct('fam',fam,'pc',pc,'a1',a,'a2',b,'box',box,'scale',sc);end,end,end,end,end
end
function m=fitM(R,y,idx,c)
views=getviews(R,c.fam);P=cell(size(views));Z=[];for v=1:numel(views),[z,P{v}]=fitPCA(views{v}(idx,:),c.pc);if v==2,z=c.a1*z;elseif v==3,z=c.a2*z;end;Z=[Z z];end %#ok<AGROW>
core=fitcsvm(Z,y(idx),'KernelFunction','rbf','BoxConstraint',c.box,'KernelScale',c.scale,'Standardize',false,'ClassNames',[1 2]);m=struct('c',c,'P',{P},'core',core);
end
function p=predictM(m,R,idx)
views=getviews(R,m.c.fam);Z=[];for v=1:numel(views),z=applyPCA(views{v}(idx,:),m.P{v});if v==2,z=m.c.a1*z;elseif v==3,z=m.c.a2*z;end;Z=[Z z];end %#ok<AGROW>
p=predict(m.core,Z);
end
function V=getviews(R,f),if f<=3,V={R{f}};elseif f==4,V=R{4};else,V=R{5};end,end
function O=subsetReps(R,idx)
O=R;for i=1:3,O{i}=R{i}(idx,:);end;for i=4:5,V=R{i};for j=1:numel(V),V{j}=V{j}(idx,:);end;O{i}=V;end
end
function [Z,p]=fitPCA(X,n)
mu=mean(X,1);sd=std(X,0,1);keep=sd>1e-10;mu=mu(keep);sd=sd(keep);X=(X(:,keep)-mu)./sd;[co,s]=pca(X,'NumComponents',min([n,size(X,1)-1,size(X,2)]));zm=mean(s);zs=std(s);zs(zs<eps)=1;Z=(s-zm)./zs;p=struct('keep',keep,'mu',mu,'sd',sd,'co',co,'zm',zm,'zs',zs);
end
function Z=applyPCA(X,p),Z=((((X(:,p.keep)-p.mu)./p.sd)*p.co)-p.zm)./p.zs;end
function X=snv(X),X=(X-mean(X,2))./max(std(X,0,2),eps);end
function v=bacc(y,p),v=.5*(mean(p(y==1)==1)+mean(p(y==2)==2));end
function v=macroF1(y,p),f=zeros(2,1);for c=1:2,tp=sum(y==c&p==c);fp=sum(y~=c&p==c);fn=sum(y==c&p~=c);f(c)=2*tp/max(2*tp+fp+fn,1);end;v=mean(f);end
