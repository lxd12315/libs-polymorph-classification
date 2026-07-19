function compare_overlap_aware_classifiers(projectRoot)
%COMPARE_OVERLAP_AWARE_CLASSIFIERS Fair comparison on identical fixed splits.
if nargin<1, projectRoot=fileparts(fileparts(fileparts(mfilename('fullpath')))); end
rng(20260718,'twister'); set(groot,'defaultFigureVisible','off');
repDir=fullfile(projectRoot,'reports'); outDir=fullfile(projectRoot,'figures','preview_20260718');
if ~isfolder(outDir),mkdir(outDir);end
S=load(fullfile(projectRoot,'data','classification','julei.mat'),'spec_al','spec_al2');
W=load(fullfile(projectRoot,'data','classification','w.mat'),'w');
X=S.spec_al(1:12286,1:37)'; Xrob=S.spec_al2(1:12286,1:37)';
[~,X]=overlap_fuse_spectra(W.w,X,0.06); [~,Xrob]=overlap_fuse_spectra(W.w,Xrob,0.06);
y=[ones(24,1);2*ones(13,1)]; yrob=y;
split=readtable(fullfile(repDir,'fixed_sample_split.csv')); im=logical(split.Modeling_set); it=logical(split.Final_test_set);

names={'PCA-LDA';'PLS-DA';'PCA-Random forest';'PCA-RBF SVM'};
repr={'SNV + PCA';'SNV + latent variables';'SNV + PCA';'SNV + PCA'};
families={'lda','pls','rf','svm'}; cvm=zeros(4,1); cvs=zeros(4,1); hm=zeros(4,3); rm=zeros(4,3); params=cell(4,1);
predHold=cell(6,1); predRobAll=cell(6,1);
for f=1:4
    C=configs(families{f}); scores=zeros(numel(C),10); q=0;
    for rep=1:2
        rng(20260718+rep,'twister'); cv=cvpartition(y(im),'KFold',5);
        for k=1:5
            q=q+1; tr=training(cv,k); va=test(cv,k); Xm=X(im,:); ym=y(im);
            for c=1:numel(C)
                mdl=fitOne(Xm(tr,:),ym(tr),C(c)); p=predictOne(mdl,Xm(va,:));
                scores(c,q)=balancedAccuracy(ym(va),p);
            end
        end
    end
    mu=mean(scores,2); sd=std(scores,0,2); [~,ib]=max(mu-0.05*sd); best=C(ib);
    mdl=fitOne(X(im,:),y(im),best); pt=predictOne(mdl,X(it,:));
    deployMdl=fitOne(X,y,best); pr=predictOne(deployMdl,Xrob);
    predHold{f}=pt; predRobAll{f}=pr;
    cvm(f)=mu(ib); cvs(f)=sd(ib); hm(f,:)=metricTriple(y(it),pt); rm(f,:)=metricTriple(yrob,pr);
    params{f}=configText(best);
end

% Proposed method is produced by the same fixed split after overlap fusion.
F=load(fullfile(repDir,'frontier_model_preview.mat'),'cvMean','cvSD','testMetrics','robMetrics');
names{5}='Overlap-aware HYDRA fusion'; repr{5}='SNV-PCA + competing convolution';
cvm(5,1)=F.cvMean(3); cvs(5,1)=F.cvSD(3); hm(5,:)=F.testMetrics(3,:); rm(5,:)=F.robMetrics(3,:); params{5}='Nested CV fusion weight, C and kernel scale';
Q=load(fullfile(repDir,'frontier_model_preview.mat'),'predTest','predRob'); predHold{5}=Q.predTest{3}; predRobAll{5}=Q.predRob{3}(1:37);

% Proximity-style class-balanced spectral prototype classifier. The number
% of neighbors per class is fixed by the minority-class count in the
% training fold, not by the external robustness labels.
protoCV=zeros(10,1); q=0; Xm=X(im,:); ym=y(im);
for rep=1:2
    rng(20260718+rep,'twister'); cv=cvpartition(ym,'KFold',5);
    for k=1:5
        q=q+1; tr=training(cv,k); va=test(cv,k);
        pp=prototypePredict(Xm(tr,:),ym(tr),Xm(va,:)); protoCV(q)=balancedAccuracy(ym(va),pp);
    end
end
names{6}='Overlap-aware proximity prototype'; repr{6}='Class-balanced spectral-angle neighborhoods';
params{6}='q = minority-class count in each training fold'; cvm(6,1)=mean(protoCV);cvs(6,1)=std(protoCV);
predHold{6}=prototypePredict(X(im,:),y(im),X(it,:)); predRobAll{6}=prototypePredict(X,y,Xrob);
hm(6,:)=metricTriple(y(it),predHold{6});rm(6,:)=metricTriple(yrob,predRobAll{6});

T=table(names,repr,params,cvm,cvs,hm(:,1),hm(:,2),hm(:,3),rm(:,1),rm(:,2),rm(:,3), ...
    'VariableNames',{'Model','Representation','Selected_parameters','CV_BAcc_mean','CV_BAcc_SD', ...
    'Holdout_Acc','Holdout_BAcc','Holdout_macroF1','Robust_Acc','Robust_BAcc','Robust_macroF1'});
writetable(T,fullfile(repDir,'Table_model_comparison_overlap_aware.csv'));
save(fullfile(repDir,'traditional_model_predictions.mat'),'predHold','predRobAll','y','yrob','im','it','names');
drawTable(T,fullfile(outDir,'Preview_04_model_comparison_table.png'));
end

function C=configs(f)
q=0; C=struct('family',{},'npc',{},'aux1',{},'aux2',{});
switch f
    case 'lda'
        for pc=[2 3 5 8], for g=[0 0.2 0.5], q=q+1;C(q)=mk(f,pc,g,0);end,end
    case 'pls'
        for pc=[2 3 5 8], q=q+1;C(q)=mk(f,pc,0,0);end
    case 'rf'
        for pc=[3 5 8 12], for leaf=[1 2],q=q+1;C(q)=mk(f,pc,leaf,80);end,end
    case 'svm'
        for pc=[3 5 8], for box=[0.3 1 3], for sc=[1 3],q=q+1;C(q)=mk(f,pc,box,sc);end,end,end
end
end
function c=mk(f,p,a,b),c=struct('family',f,'npc',p,'aux1',a,'aux2',b);end

function model=fitOne(X,y,c)
Xs=rowSNV(X);
if strcmp(c.family,'pls')
    sd=std(Xs,0,1); keep=sd>1e-10; mu=mean(Xs(:,keep),1); sd=sd(keep); Z=(Xs(:,keep)-mu)./sd;
    yy=2*(y==2)-1; [~,~,~,~,beta]=plsregress(Z,yy,c.npc);
    model=struct('c',c,'keep',keep,'mu',mu,'sd',sd,'beta',beta); return
end
[Z,p]=fitPCA(Xs,c.npc);
switch c.family
    case 'lda', core=fitcdiscr(Z,y,'DiscrimType','linear','Gamma',c.aux1,'ClassNames',[1 2]);
    case 'rf'
        core=TreeBagger(c.aux2,Z,y,'Method','classification','MinLeafSize',c.aux1, ...
            'NumPredictorsToSample',max(1,round(sqrt(size(Z,2)))),'OOBPrediction','off');
    case 'svm', core=fitcsvm(Z,y,'KernelFunction','rbf','BoxConstraint',c.aux1,'KernelScale',c.aux2, ...
            'Standardize',false,'ClassNames',[1 2]);
end
model=struct('c',c,'p',p,'core',core);
end

function pred=predictOne(model,X)
Xs=rowSNV(X);
if strcmp(model.c.family,'pls')
    Z=(Xs(:,model.keep)-model.mu)./model.sd; score=[ones(size(Z,1),1) Z]*model.beta;
    pred=ones(size(score)); pred(score>=0)=2; return
end
Z=applyPCA(Xs,model.p);
if strcmp(model.c.family,'rf')
    pred=str2double(predict(model.core,Z));
else, pred=predict(model.core,Z); end
end

function pred=prototypePredict(Xtrain,ytrain,Xquery)
Xtrain=rowSNV(Xtrain);Xquery=rowSNV(Xquery);D=pdist2(Xquery,Xtrain,'cosine');
classes=[1 2];q=min([sum(ytrain==1),sum(ytrain==2)]);score=zeros(size(D,1),2);
for c=classes
    Dc=sort(D(:,ytrain==c),2,'ascend');score(:,c)=mean(Dc(:,1:q),2);
end
[~,pred]=min(score,[],2);
end

function [Z,p]=fitPCA(X,npc)
mu=mean(X,1); sd=std(X,0,1); keep=sd>1e-10; mu=mu(keep);sd=sd(keep); X=(X(:,keep)-mu)./sd;
[coef,score]=pca(X,'NumComponents',min([npc,size(X,1)-1,size(X,2)])); zm=mean(score,1);zs=std(score,0,1);zs(zs<eps)=1;
Z=(score-zm)./zs;p=struct('keep',keep,'mu',mu,'sd',sd,'coef',coef,'zm',zm,'zs',zs);
end
function Z=applyPCA(X,p),Z=((((X(:,p.keep)-p.mu)./p.sd)*p.coef)-p.zm)./p.zs;end
function X=rowSNV(X),X=(X-mean(X,2))./max(std(X,0,2),eps);end
function m=metricTriple(y,p),m=[mean(y==p),balancedAccuracy(y,p),macroF1(y,p)];end
function v=balancedAccuracy(y,p),c=unique(y(:))';a=zeros(size(c));for i=1:numel(c),a(i)=mean(p(y==c(i))==c(i));end;v=mean(a);end
function v=macroF1(y,p),c=unique(y(:))';f=zeros(size(c));for i=1:numel(c),z=c(i);tp=sum(y==z&p==z);fp=sum(y~=z&p==z);fn=sum(y==z&p~=z);f(i)=2*tp/max(2*tp+fp+fn,1);end;v=mean(f);end
function s=configText(c)
switch c.family
    case 'lda',s=sprintf('%d PCs; gamma %.1f',c.npc,c.aux1);
    case 'pls',s=sprintf('%d latent variables',c.npc);
    case 'rf',s=sprintf('%d PCs; leaf %d; %d trees',c.npc,c.aux1,c.aux2);
    case 'svm',s=sprintf('%d PCs; C %.1f; scale %.1f',c.npc,c.aux1,c.aux2);
end
end
function drawTable(T,path)
fig=figure('Color','w','Units','pixels','Position',[20 20 1560 500]);ax=axes(fig,'Position',[0.015 0.08 0.97 0.84]);axis(ax,[0 1 0 1]);axis(ax,'off');hold(ax,'on');
headers={'Model','CV BAcc','Holdout Acc','Holdout BAcc','Robust Acc','Robust BAcc','Robust F1'};
width=[0.28 0.12 0.12 0.13 0.12 0.12 0.11]; x=[0 cumsum(width)]; rh=0.13; top=0.90;
for j=1:numel(headers),rectangle(ax,'Position',[x(j),top,width(j),rh],'FaceColor',[0.04 0.22 0.62],'EdgeColor','w');text(ax,x(j)+width(j)/2,top+rh/2,headers{j},'Color','w','FontWeight','bold','HorizontalAlignment','center','FontName','Times New Roman','FontSize',12);end
for i=1:height(T)
    y=top-i*rh; bg=[1 1 1]; if mod(i,2)==0,bg=[0.94 0.96 0.98];end;if i==height(T),bg=[1.00 0.90 0.97];end
    vals={T.Model{i},sprintf('%.1f %s %.1f',100*T.CV_BAcc_mean(i),char(177),100*T.CV_BAcc_SD(i)),sprintf('%.1f',100*T.Holdout_Acc(i)), ...
        sprintf('%.1f',100*T.Holdout_BAcc(i)),sprintf('%.1f',100*T.Robust_Acc(i)),sprintf('%.1f',100*T.Robust_BAcc(i)),sprintf('%.1f',100*T.Robust_macroF1(i))};
    for j=1:numel(headers),rectangle(ax,'Position',[x(j),y,width(j),rh],'FaceColor',bg,'EdgeColor',[0.72 0.72 0.72]);ha='center';tx=x(j)+width(j)/2;if j==1,ha='left';tx=x(j)+0.012;end;text(ax,tx,y+rh/2,vals{j},'HorizontalAlignment',ha,'FontName','Times New Roman','FontSize',12,'FontWeight',ternary(i==height(T),'bold','normal'));end
end
text(ax,0,0.02,'All values are percentages. Hyperparameters were selected only within the modeling set by repeated stratified CV.', ...
    'FontName','Times New Roman','FontSize',11,'FontAngle','italic');
exportgraphics(fig,path,'Resolution',300);close(fig);
end
function z=ternary(tf,a,b),if tf,z=a;else,z=b;end,end
