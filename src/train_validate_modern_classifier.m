function train_validate_modern_classifier(projectRoot)
%TRAIN_VALIDATE_MODERN_CLASSIFIER Leakage-free small-sample spectral classifier.
% Outer 50% holdout is untouched during tuning. Repeated stratified CV within
% the modeling half selects PCA dimension and RBF-SVM hyperparameters.
if nargin < 1
    projectRoot=fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
set(groot,'defaultFigureVisible','off'); rng(20260717,'twister');
outDir=fullfile(projectRoot,'figures','optimized'); reportDir=fullfile(projectRoot,'reports');
if ~exist(outDir,'dir'),mkdir(outDir);end

S=load(fullfile(projectRoot,'data','classification','julei.mat'));
X=S.spec_al(1:12286,1:37)'; y=[ones(24,1);2*ones(13,1)];
Xrob=S.spec_al2(1:12286,:)';

outer=cvpartition(y,'Holdout',0.50);
idxModel=training(outer); idxTest=test(outer);
Xmodel=X(idxModel,:); ymodel=y(idxModel); Xtest=X(idxTest,:); ytest=y(idxTest);

pcs=[3 5 8 12]; Cs=[0.1 1 10]; scales=[0.3 1 3];
records=[]; bestScore=-inf; best=[pcs(1),Cs(1),scales(1)];
for np=pcs
    for C=Cs
        for sc=scales
            foldScores=zeros(15,1); q=0;
            for rep=1:3
                rng(20260717+rep,'twister'); cv=cvpartition(ymodel,'KFold',5);
                for k=1:cv.NumTestSets
                    q=q+1; tr=training(cv,k); va=test(cv,k);
                    [Ztr,prep]=fitPreprocess(Xmodel(tr,:),np);
                    Zva=applyPreprocess(Xmodel(va,:),prep);
                    mdl=fitcsvm(Ztr,ymodel(tr),'KernelFunction','rbf','BoxConstraint',C, ...
                        'KernelScale',sc,'Standardize',false,'ClassNames',[1 2]);
                    phat=predict(mdl,Zva);
                    foldScores(q)=balancedAccuracy(ymodel(va),phat);
                end
            end
            score=mean(foldScores); records=[records;np C sc score std(foldScores)]; %#ok<AGROW>
            if score>bestScore, bestScore=score; best=[np C sc]; end
        end
    end
end

[Zmodel,prep]=fitPreprocess(Xmodel,best(1)); Ztest=applyPreprocess(Xtest,prep);
mdl=fitcsvm(Zmodel,ymodel,'KernelFunction','rbf','BoxConstraint',best(2), ...
    'KernelScale',best(3),'Standardize',false,'ClassNames',[1 2]);
[predTest,scoreTest]=predict(mdl,Ztest);
Zrob=applyPreprocess(Xrob,prep); [predRob,scoreRob]=predict(mdl,Zrob);

cm=confusionmat(ytest,predTest,'Order',[1 2]);
testBal=balancedAccuracy(ytest,predTest); testAcc=mean(ytest==predTest);
testF1=macroF1(ytest,predTest);
robTruth=[ones(24,1);2*ones(13,1)];
robKnownPred=predRob(1:37); robBal=balancedAccuracy(robTruth,robKnownPred);
robAcc=mean(robTruth==robKnownPred); robF1=macroF1(robTruth,robKnownPred);

% Open-set flag calibrated only from modeling-sample margins (5th percentile).
[~,scoreModel]=predict(mdl,Zmodel); modelMargin=abs(scoreModel(:,2)-scoreModel(:,1));
openThreshold=prctile(modelMargin,5);
robMargin=abs(scoreRob(:,2)-scoreRob(:,1)); unknownFlag=robMargin<openThreshold;

metrics=table(best(1),best(2),best(3),bestScore,testAcc,testBal,testF1,robAcc,robBal,robF1, ...
    openThreshold,predRob(38),robMargin(38),unknownFlag(38),sum(idxModel),sum(idxTest), ...
    'VariableNames',{'PCA_components','BoxConstraint','KernelScale','InnerCV_balanced_accuracy', ...
    'Holdout_accuracy','Holdout_balanced_accuracy','Holdout_macroF1','Robustness_accuracy', ...
    'Robustness_balanced_accuracy','Robustness_macroF1','Open_set_margin_threshold', ...
    'Sample38_predicted_known_class','Sample38_margin','Sample38_flagged_unknown','Modeling_n','Final_test_n'});
writetable(metrics,fullfile(reportDir,'optimized_model_metrics.csv'));
tuning=array2table(records,'VariableNames',{'PCA_components','BoxConstraint','KernelScale','Mean_balanced_accuracy','SD_balanced_accuracy'});
writetable(tuning,fullfile(reportDir,'hyperparameter_validation.csv'));
assign=table((1:37)',y,idxModel,idxTest,'VariableNames',{'Sample','Class','Modeling_set','Final_test_set'});
writetable(assign,fullfile(reportDir,'fixed_sample_split.csv'));

blue=[0.12 0.47 0.71]; red=[0.84 0.15 0.16]; green=[0.17 0.63 0.17];
f=figure('Color','w','Position',[80 80 1450 820]); tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
nexttile; stem(find(idxTest),predTest,'filled','Color',blue); hold on; plot(find(idxTest),ytest,'o','Color',red,'MarkerFaceColor',red);
xlabel('Final-test sample');ylabel('Class');yticks([1 2]);yticklabels({'A','B'});legend({'Prediction','Truth'},'Location','best');localAxes(gca);
nexttile; imagesc(cm);axis image;colorbar;xticks(1:2);yticks(1:2);xticklabels({'A','B'});yticklabels({'A','B'});xlabel('Predicted');ylabel('Actual');localAxes(gca);
for i=1:2,for j=1:2,text(j,i,num2str(cm(i,j)),'HorizontalAlignment','center','Color','w','FontWeight','bold');end,end
nexttile; bar(100*[testAcc testBal testF1 robAcc robBal robF1],'FaceColor',green);ylim([0 100]);ylabel('Score (%)');
xticklabels({'Holdout Acc','Holdout BAcc','Holdout F1','Robust Acc','Robust BAcc','Robust F1'});xtickangle(25);localAxes(gca);
nexttile; bar(robMargin,'FaceColor',blue);hold on;yline(openThreshold,'--','Color',red,'LineWidth',1.4);xlabel('Robustness sample');ylabel('Absolute SVM margin');
legend({'Margin','Open-set threshold'},'Location','best');localAxes(gca);
exportgraphics(f,fullfile(outDir,'Fig_leakage_free_SVM_results.png'),'Resolution',300);close(f);
save(fullfile(reportDir,'optimized_model.mat'),'mdl','prep','best','idxModel','idxTest','predTest','scoreTest','predRob','scoreRob','unknownFlag','openThreshold');
end

function [Z,p]=fitPreprocess(X,npc)
rowMean=mean(X,2); rowStd=std(X,0,2); rowStd(rowStd<eps)=1;
Xs=(X-rowMean)./rowStd;
mu=mean(Xs,1); sd=std(Xs,0,1); keep=sd>1e-10;
Xk=Xs(:,keep)-mu(keep);
[coeff,score]=pca(Xk,'NumComponents',min([npc,size(Xk,1)-1,size(Xk,2)]));
zmu=mean(score,1); zsd=std(score,0,1); zsd(zsd<eps)=1;
Z=(score-zmu)./zsd;
p=struct('keep',keep,'mu',mu(keep),'coeff',coeff,'zmu',zmu,'zsd',zsd);
end

function Z=applyPreprocess(X,p)
rs=std(X,0,2);rs(rs<eps)=1;Xs=(X-mean(X,2))./rs;
score=(Xs(:,p.keep)-p.mu)*p.coeff;Z=(score-p.zmu)./p.zsd;
end

function v=balancedAccuracy(y,p)
v=0.5*(mean(p(y==1)==1)+mean(p(y==2)==2));
end

function v=macroF1(y,p)
f=zeros(2,1);for c=1:2,tp=sum((y==c)&(p==c));fp=sum((y~=c)&(p==c));fn=sum((y==c)&(p~=c));f(c)=2*tp/max(2*tp+fp+fn,1);end;v=mean(f);
end

function localAxes(ax)
set(ax,'FontName','Arial','FontSize',11,'LineWidth',1,'Box','on','TickDir','out');
end
