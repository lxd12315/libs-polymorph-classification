function finalize_vip_feature_attribution(projectRoot)
if nargin<1,projectRoot=fileparts(fileparts(fileparts(mfilename('fullpath'))));end
set(groot,'defaultFigureVisible','off');
outDir=fullfile(projectRoot,'figures','optimized');reportDir=fullfile(projectRoot,'reports');
S=load(fullfile(projectRoot,'data','classification','julei.mat'));Wv=load(fullfile(projectRoot,'data','classification','w.mat'));
X=S.spec_al(1:12286,1:37)';Y=[zeros(24,1);ones(13,1)];w=Wv.w(:)';
wl=[w(1:2023),w(2065:4076),w(4113:6124),w(6161:8166),w(8226:12255)];wl=[wl,zeros(1,12286-numel(wl))];
[T,W,Wo]=localOpls(X,Y,1);vip=localVip(X,Y,T,W,Wo);
idx=S.significant_peaksCopy(:);featW=wl(idx)';featVip=vip(idx);m0=mean(X(Y==0,idx),1)';m1=mean(X(Y==1,idx),1)';assoc=1+(m1>m0);

names={'C I','H I','N I','O I','CN','C2','NH','Fe','Mg','Ca','Na','K','Ar I','Ar II'};
peaks={ [193.09 247.86], [486.10 656.28], [742.36 744.23 746.83 818.80 821.63 868.03 870.32 871.17], ...
 [715.67 777.17 777.42 822.18], [357.3:0.05:360 376:0.05:388.8 414.03:0.05:422], ...
 [465.4:0.05:474.2 511.7:0.05:517.2 549.6:0.05:564.1],334.5:0.05:337.8,[248.33 358.12], ...
 [279.55 285.21],[422.67 393.37 396.85],[588.59 588.99],[766.49 769.90], ...
 [696.54 706.72 750.39 763.51 811.53 842.46],[434.81 480.60 487.99]};
tol=0.3;weightSum=zeros(numel(names),1);count=zeros(numel(names),1);assignment=strings(numel(idx),1);
for i=1:numel(idx)
    hit=[];
    for j=1:numel(names)
        if any(abs(peaks{j}-featW(i))<=tol)
            weightSum(j)=weightSum(j)+featVip(i);count(j)=count(j)+1;hit=[hit,string(names{j})]; %#ok<AGROW>
        end
    end
    if isempty(hit),assignment(i)="Unassigned";else,assignment(i)=strjoin(hit,'/');end
end
featureTable=table(idx,featW,featVip,m0,m1,assoc,assignment,'VariableNames', ...
    {'Feature_index','Wavelength_nm','VIP','Mean_class_A','Mean_class_B','Stronger_class','Attribution'});
writetable(featureTable,fullfile(reportDir,'Table_X_feature_level.csv'));
summary=table(string(names(:)),weightSum,count,'VariableNames',{'Attribution','VIP_weight_sum','Matched_feature_count'});
writetable(summary,fullfile(reportDir,'Table_X_attribution_summary.csv'));

blue=[0.12 0.47 0.71];red=[0.84 0.15 0.16];green=[0.17 0.63 0.17];
f=figure('Color','w','Position',[50 50 1550 1050]);tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
nexttile;hold on;plot(wl(1:12083),mean(X(Y==0,1:12083),1),'Color',blue,'LineWidth',1);plot(wl(1:12083),mean(X(Y==1,1:12083),1),'Color',red,'LineWidth',1);
scatter(featW,m0,18,'k','filled');scatter(featW,m1,18,green,'filled');xlabel('Wavelength (nm)');ylabel('Normalized intensity (a.u.)');legend({'Class A','Class B','Selected: A','Selected: B'},'Location','northeast');grid on;localAxes(gca);
nexttile;hold on;plot(wl(1:12083),vip(1:12083),'k','LineWidth',0.8);yline(4,'--','Color',red,'LineWidth',1.2);scatter(featW,featVip,24,blue,'filled');xlabel('Wavelength (nm)');ylabel('VIP score');legend({'VIP','VIP=4','Stored 36-feature panel'},'Location','northeast');grid on;localAxes(gca);
nexttile;hold on;[~,ord]=sort(featW);sv=featVip/max(featVip);n=numel(idx);
for q=1:n,i=ord(q);target=0.75+0.18*(assoc(i)==1);col=red;if assoc(i)==1,col=blue;end;col=0.55*[1 1 1]+0.45*col;plot([0.25 0.75],[q target*n],'-','Color',col,'LineWidth',0.5+5*sv(i));text(0.23,q,sprintf('%.2f',featW(i)),'HorizontalAlignment','right','FontSize',7);end
text(0.79,0.93*n,'Polymorph A','Color',blue,'FontWeight','bold');text(0.79,0.75*n,'Polymorph B','Color',red,'FontWeight','bold');xlim([0 1]);ylim([0 n+1]);axis off;title('Feature–polymorph association');
nexttile;keep=weightSum>0;bar(categorical(names(keep)),weightSum(keep),'FaceColor',green);ylabel('Summed VIP weight');xtickangle(30);localAxes(gca);
exportgraphics(f,fullfile(outDir,'Fig_VIP_feature_attribution_4panel.png'),'Resolution',300);close(f);
end

function [T,W,Wo]=localOpls(X,Y,n)
Xc=X-mean(X,1);Yc=Y-mean(Y);W=(Yc'*Xc)'/norm(Yc'*Xc);T=Xc*W;p=(T'*Xc)'/(T'*T);Wo=zeros(size(X,2),n);
for k=1:n,Xr=Xc-T*p';wo=(T'*Xr)'/(T'*T);wo=wo/norm(wo);to=Xr*wo;po=(to'*Xr)'/(to'*to);Wo(:,k)=wo;Xc=Xr-to*po';end
W=(Yc'*Xc)'/norm(Yc'*Xc);T=Xc*W;
end
function vip=localVip(X,Y,T,W,Wo)
p=size(X,2);ssy=sum((Y-mean(Y)).^2);pred=sum((T*pinv(T)*Y).^2);vip=sqrt(p*(W.^2+sum(Wo.^2,2))*pred/ssy);
end
function localAxes(ax),set(ax,'FontName','Arial','FontSize',10,'LineWidth',1,'Box','on','TickDir','out');end
