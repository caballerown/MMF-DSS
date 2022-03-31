% Caballero, Gaw, Jenkins, and Johnstone
% Toward Automated Instructor Pilots in Legacy Air Force Systems: 
% Physiology-based Flight Difficulty Classification via Machine Learning
% Testing

%% Preliminaries
% random number seed and clearing
clc
clear 
close all
rng(42)

% load in cleaned data containting only physiological features
xtrain= readtable('xtrain.csv','VariableNamingRule','preserve');
ytrain= readtable('ytrain.csv','VariableNamingRule','preserve');
xtest= readtable('xtest.csv','VariableNamingRule','preserve');
ytest= readtable('ytest.csv','VariableNamingRule','preserve');

% %Uncomment to remove non-ETK features (i.e., eye-tracking only models)
% xtrain(:,2) = [];
% xtrain(:,7) = [];
% xtest(:,2) = [];
% xtest(:,7) = [];

% %Uncomment to load in cleaned data containing all features (i.e., both physiological and
% non-physiological (aircraft)).  
% xtrain= readtable('xtrain2.csv','VariableNamingRule','preserve');
% ytrain= readtable('ytrain2.csv','VariableNamingRule','preserve');
% xtest= readtable('xtest2.csv','VariableNamingRule','preserve');
% ytest= readtable('ytest2.csv','VariableNamingRule','preserve');

% remove one sample in the test set to even out classes
xtest(13,:)=[];
ytest(13,:)=[];

% standardize the data
xmean = mean(xtrain{:,1:end},1);
xstd = std(xtrain{:,1:end},1);
xtrain_n = xtrain;
xtest_n = xtest;
xtrain_n{:,1:end} = (xtrain{:,1:end} - xmean)./xstd;
xtest_n{:,1:end}= (xtest{:,1:end} - xmean)./xstd;

%% AdaBoostM1 testing
% https://www.mathworks.com/help/stats/framework-for-ensemble-learning.html#bsxx7ah
% https://www.mathworks.com/help/stats/fitcensemble.html
ada = [];
for i = .01:.01:1 % learn rate - default is 1
    for j = 10:10:200 % number of cycles - default is 100
        Mdl = fitcensemble([xtrain_n, ytrain],"Difficulty",'Method','AdaBoostM1',...
            'NumLearningCycles',j,...
            'Learners','tree','LearnRate',i); %Tree/Discriminant  %1000,0.03 87.3
        testAccuracy = 1 - loss(Mdl,xtest_n,ytest);
        ada = [ada; i,j,testAccuracy];
    end
end

%% SVM testing
% https://www.mathworks.com/help/stats/fitcsvm.html#bt8v_23-1
kf = [{'rbf', 'linear', 'polynomial'}];
bc = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000];
svm= [];
for i = 1:3 % kernel function - default is linear
    for j = bc % box constraint (C-penalty) - default is 1
        if i ==3
            for k =2:4
                Mdl = fitcsvm([xtrain_n, ytrain],"Difficulty","KernelFunction",char(kf(i)),"BoxConstraint",j,"PolynomialOrder",k);
                testAccuracy = 1 - loss(Mdl,xtest_n,ytest);
                svm = [svm; i,j,k,testAccuracy];
            end
        else
            Mdl = fitcsvm([xtrain_n, ytrain],"Difficulty","KernelFunction",char(kf(i)),"BoxConstraint",j);
            testAccuracy = 1 - loss(Mdl,xtest_n,ytest);
            svm = [svm; i,j,0,testAccuracy];
        end
    end
end

%% kNN testing
% https://www.mathworks.com/help/stats/fitcknn.html#bt6cr9l-2
dst = [{'chebychev','euclidean','minkowski','hamming'}];
wgt = [{'equal','inverse','squaredinverse'}];
knn= [];
for i = 1:70 % k - default is 1
    for j = 1:4 % dst - default is hamming
        for k = 1:3 % dst wgt - default is equal
            Mdl = fitcknn([xtrain_n, ytrain],"Difficulty",'NumNeighbors',i,'Distance',char(dst(j)),'DistanceWeight',char(wgt(k)));
            testAccuracy = 1 - loss(Mdl,xtest_n,ytest);
            knn = [knn; i,j,k,testAccuracy];
        end
    end
end

%% NN testing
% https://www.mathworks.com/help/stats/fitcnet.html#mw_70e82a26-9004-467b-bba0-43b84468fe66
act = [{'none','relu','sigmoid','tanh'}];
lambda = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3];
nn= [];
for i = 5:5:50 % layer size - default 10
    for j = 1:4 % activation - default relu
        for k = lambda % regularization - default 0
            for l = 1:3 % num layers; default 1
                if l == 1
                    Mdl = fitcnet([xtrain_n, ytrain],"Difficulty","LayerSizes",[i],...
                        "Activations",char(act(j)),"lambda",k);
                    testAccuracy = 1 - loss(Mdl,xtest_n,ytest);
                    nn = [nn; i,j,k,l,testAccuracy];
                elseif l ==2
                    Mdl = fitcnet([xtrain_n, ytrain],"Difficulty","LayerSizes",[i,i],...
                        "Activations",char(act(j)),"lambda",k);
                    testAccuracy = 1 - loss(Mdl,xtest_n,ytest);
                    nn = [nn; i,j,k,l,testAccuracy];
                else
                    Mdl = fitcnet([xtrain_n, ytrain],"Difficulty","LayerSizes",[i,i,i],...
                        "Activations",char(act(j)),"lambda",k);
                    testAccuracy = 1 - loss(Mdl,xtest_n,ytest);
                    nn = [nn; i,j,k,l,testAccuracy];
                end
            end
        end
    end
end

%% Box Plot
% https://www.mathworks.com/help/stats/boxplot.html
% https://www.mathworks.com/matlabcentral/answers/60818-boxplot-with-vectors-of-different-lengths
A1 = ada(:,3)';
A2 = nn(:,5)';
A3 = knn(:,4)';
A4 = svm(:,4)';
G = [zeros(size(A1))  ones(size(A2)) 2*ones(size(A3)) 3*ones(size(A4))];
X = [A1, A2, A3, A4];
boxplot(X,G,'Labels',{'AdaBoost','NN','kNN','SVM'});
xlabel('Technique')
ylabel('Accuracy')

%% sort results
ada = sortrows(ada,3,"descend");
svm = sortrows(svm,4,"descend");
knn = sortrows(knn,4,"descend");
nn = sortrows(nn,5,"descend");

%% train best models
Mdl_ada = fitcensemble([xtrain_n, ytrain],"Difficulty",'Method','AdaBoostM1',...
            'NumLearningCycles',ada(1,2),...
            'Learners','tree','LearnRate',ada(1,1));
cm_ada = confusionmat(ytest.Difficulty,predict(Mdl_ada,xtest_n));

Mdl_svm =fitcsvm([xtrain_n, ytrain],"Difficulty","KernelFunction",'linear',"BoxConstraint",0.0001);
cm_svm = confusionmat(ytest.Difficulty,predict(Mdl_svm,xtest_n));

Mdl_knn = fitcknn([xtrain_n, ytrain],"Difficulty",'NumNeighbors',31,'Distance','euclidean','DistanceWeight','equal');
cm_knn = confusionmat(ytest.Difficulty,predict(Mdl_knn,xtest_n));

Mdl_nn = fitcnet([xtrain_n, ytrain],"Difficulty","LayerSizes",[5],...
                        "Activations",'tanh',"lambda",.01);
cm_nn = confusionmat(ytest.Difficulty,predict(Mdl_nn,xtest_n));

%% AUC and ROC
% https://www.mathworks.com/help/stats/perfcurve.html

%ada
[~,score_ada] = predict(Mdl_ada,xtest_n);
[Xada,Yada,Tada,AUCada] = perfcurve(ytest.Difficulty,score_ada(:,2),'4');

%svm
[~,score_svm] = predict(Mdl_svm,xtest_n);
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(ytest.Difficulty,score_svm(:,2),'4');

%knn
[~,score_knn] = predict(Mdl_knn,xtest_n);
[Xknn,Yknn,Tknn,AUCknn] = perfcurve(ytest.Difficulty,score_knn(:,2),'4');

%nn
[~,score_nn] = predict(Mdl_nn,xtest_n);
[Xnn,Ynn,Tnn,AUCnn] = perfcurve(ytest.Difficulty,score_nn(:,2),'4');

% AUC and ROC figure
figure
plot(Xada,Yada)
hold on
txt = ['AUC: ' num2str(AUCada)];
annotation('textbox', [0.65, 0.1, 0.1, 0.1], 'String', "AUC = " + AUCada)
line([0,1],[0,1],'linestyle',':','color','k')
xlabel('False positive rate'); ylabel('True positive rate');
hold off


%% Best-tuned model metrics
%ada
ada_recall = cm_ada(2,2)/(cm_ada(2,1)+cm_ada(2,2));
ada_precision = cm_ada(2,2)/(cm_ada(1,2)+cm_ada(2,2));
ada_f1 = (2*ada_precision*ada_recall)/(ada_precision+ada_recall);

ada_recall1 = cm_ada(1,1)/(cm_ada(1,2)+cm_ada(1,1));
ada_precision1 = cm_ada(1,1)/(cm_ada(2,1)+cm_ada(1,1));
ada_f11 = (2*ada_precision1*ada_recall1)/(ada_precision1+ada_recall1);

ada_recall_wgt = (ada_recall+ ada_recall1)/2;
ada_precision_wgt = (ada_precision + ada_precision1)/2;
ada_f1_wgt = (ada_f1 + ada_f11)/2;

%svm
svm_recall = cm_svm(2,2)/(cm_svm(2,1)+cm_svm(2,2));
svm_precision = cm_svm(2,2)/(cm_svm(1,2)+cm_svm(2,2));
svm_f1 = (2*svm_precision*svm_recall)/(svm_precision+svm_recall);

svm_recall1 = cm_svm(1,1)/(cm_svm(1,2)+cm_svm(1,1));
svm_precision1 = cm_svm(1,1)/(cm_svm(2,1)+cm_svm(1,1));
svm_f11 = (2*svm_precision1*svm_recall1)/(svm_precision1+svm_recall1);

svm_recall_wgt = (svm_recall+ svm_recall1)/2;
svm_precision_wgt = (svm_precision + svm_precision1)/2;
svm_f1_wgt = (svm_f1 + svm_f11)/2;

%knn
knn_recall = cm_knn(2,2)/(cm_knn(2,1)+cm_knn(2,2));
knn_precision = cm_knn(2,2)/(cm_knn(1,2)+cm_knn(2,2));
knn_f1 = (2*knn_precision*knn_recall)/(knn_precision+knn_recall);

knn_recall1 = cm_knn(1,1)/(cm_knn(1,2)+cm_knn(1,1));
knn_precision1 = cm_knn(1,1)/(cm_knn(2,1)+cm_knn(1,1));
knn_f11 = (2*knn_precision1*knn_recall1)/(knn_precision1+knn_recall1);

knn_recall_wgt = (knn_recall+ knn_recall1)/2;
knn_precision_wgt = (knn_precision + knn_precision1)/2;
knn_f1_wgt = (knn_f1 + knn_f11)/2;

%nn
nn_recall = cm_nn(2,2)/(cm_nn(2,1)+cm_nn(2,2));
nn_precision = cm_nn(2,2)/(cm_nn(1,2)+cm_nn(2,2));
nn_f1 = (2*nn_precision*nn_recall)/(nn_precision+nn_recall);

nn_recall1 = cm_nn(1,1)/(cm_nn(1,2)+cm_nn(1,1));
nn_precision1 = cm_nn(1,1)/(cm_nn(2,1)+cm_nn(1,1));
nn_f11 = (2*nn_precision1*nn_recall1)/(nn_precision1+nn_recall1);

nn_recall_wgt = (nn_recall+ nn_recall1)/2;
nn_precision_wgt = (nn_precision + nn_precision1)/2;
nn_f1_wgt = (nn_f1 + nn_f11)/2;

%% Best-tuned AdaBoost Classifier Output by True Difficulty Level
Mdl_ada2 = fitcensemble([xtrain_n, ytrain],"Difficulty",'Method','AdaBoostM1',...
            'NumLearningCycles',ada(1,2),...
            'Learners','tree','LearnRate',ada(1,1),'ScoreTransform','logit');
[~,score_ada2] = predict(Mdl_ada2,xtest_n);
figure
low = (ytest.Difficulty == 1).*score_ada2(:,2); % probability of y = high
low(low==0)=[];
edges = [0:.2:1];
histogram(low,edges)
axis([0 1 0 25])
ylabel('Observed Frequency')
xlabel('Probability of High-Difficulty Flight Classification')
title('Low-Difficulty Flight Samples')

figure
high = (ytest.Difficulty == 4).*score_ada2(:,2) % probability of y = high
high(high==0)=[]
edges = [0:.2:1]
histogram(high,edges)
axis([0 1 0 25])
ylabel('Observed Frequency')
xlabel('Probability of High-Difficulty Flight Classification')
title('High-Difficulty Flight Samples')

%% Accuracy Curves For Training, Validation, and Testing Sets Obtained From AdaBoost Model With Learn Rate = 0.09
N = 100;
train_error= zeros(N,1);
validation_error= zeros(N,1);
test_error = zeros(N,1);
for n = 1:N
    Mdl_ada = fitcensemble([xtrain_n, ytrain],"Difficulty",'Method','AdaBoostM1',...
        'NumLearningCycles',n,...
        'Learners','tree','LearnRate',ada(1,1));
    train_error(n) = loss(Mdl_ada,xtrain_n,ytrain);
    test_error(n) = loss(Mdl_ada,xtest_n,ytest);
end
Mdl_ada = fitcensemble([xtrain_n, ytrain],"Difficulty",'Method','AdaBoostM1',...
    'NumLearningCycles',N,...
    'Learners','tree','LearnRate',ada(1,1),'KFold',5);
validation_error = kfoldLoss(Mdl_ada,'Mode','cumulative')

figure
plot(1:N, 100*(1 - train_error),'color','blue', 'LineWidth',2)
hold on 
plot(1:N, 100*(1 - validation_error),'color','green', 'LineWidth',2)
plot(1:N, 100*(1 - test_error),'color','red', 'LineWidth',2)
line([35,35],[50,100],'color','k','linestyle',':')
axis([0 50 50 100])
legend('Training','Validation','Testing','Best','Location','SouthEast')
ylabel('Accuracy (%)')
xlabel('Number of Learning Cycles')

%% SHAP values
shap_low= [];
shap_high =[];
for k=1:size(xtrain,1)
    h=figure;
    queryPoint = xtrain(k,:);
    explainer = shapley(Mdl_ada,xtrain,'QueryPoint',queryPoint,'Method','conditional-kernel');
    shap_low = [shap_low, explainer.ShapleyValues{:,2}];
    shap_high = [shap_high, explainer.ShapleyValues{:,3}];
    plot(explainer)
    saveas(h,sprintf('ShapleyQueryPoint%d.png',k)); % will create FIG1, FIG2,...
end

labels = explainer.ShapleyValues{:,1};
shap_low_avg = mean(shap_low,2)
shap_high_avg = mean(shap_high,2)
shap_high_avg2 = mean(abs(shap_high),2)
labels2 ={
    '$f^P_{23,3}$ (Diameter of Right Pupil, Third FPC)'
    '$f^S_{3,11}$ (Electrodermal activity on left hand, PosCount)' 
    '$f^S_{17,2}$ (Normalized longitudinal direction of left eye gaze, SD)'
    '$f^S_{16,8}$ (Normalized lateral direction of left eye gaze, MedAbsDev)'
    '$f^S_{23,15}$ (Diameter of right pupil, TimeMax)'
    'f^S_{20,2}$ (Normalized longitudinal direction of right eye gaze, SD)'
    '$f^S_{11,13}$ (Longitudinal axis origin of left eye gaze, Skew)'
    '$f^S_{36,11}$ (Plethysmogram measured at middle finger s tip, PosCount)'
    '$f^P_{22,3}$ (Diameter of Left Pupil, Third FPC)'   }

labels3 ={
    'f^P_{23,3}'
    'f^S_{3,11}' 
    'f^S_{17,2}'
    'f^S_{16,8}'
    'f^S_{23,15}'
    'f^S_{20,2}'
    'f^S_{11,13}'
    'f^S_{36,11}'
    'f^P_{22,3}'   }

[B I] = sortrows(shap_high_avg2)
figure
barh(B)
yticklabels(labels3(I))
xlabel('SHAP value (impact on model output)')

figure
for i = 1:9
scatter1 = scatter(shap_high(I(i),:), (i)*ones(size(xtrain,1),1),'filled','MarkerFaceColor','b');
scatter1.MarkerFaceAlpha = 0.2;
scatter1.MarkerEdgeAlpha =0.2;
hold on 
scatter(B(i), i,85,'filled','MarkerFaceColor','r','Marker','d')
end
line([0,0],[0,10],'color','k')
yticklabels([' ';labels3(I)])
xlabel('SHAP value (impact on model output)')

