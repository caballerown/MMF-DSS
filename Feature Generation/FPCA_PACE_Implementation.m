%FPCA on Multimodal Pilot Data
clear;clc;
%datasets of interest
dataset = {'stream-lslshimmereda','stream-lslshimmeremg','stream-lslshimmerrespecg',...
            'stream-lslhtcviveeye','stream-lslxp11xpcac','stream-lslxp11xpcplt'};
%particular dataset to output FPCs (number corresponds to index in dataset
%variable)
dataset_idx = 6;
dataset = dataset{dataset_idx};
%key folders in our analysis
paceFolder = 'I:\Matlab Packages\PACE_matlab-master'; %points to the PACE package
input_dev = 'I:\Research\Afwerx Datathon\dataChallenge_release\data\developmentSet\task-ils_rem_sub3_sub27r1r2'; %location of development set data
input_val = 'I:\Research\Afwerx Datathon\dataChallenge_release\data\evaluationSet\task-ils'; %location of validation data
output0 = 'I:\Research\Afwerx Datathon\dataChallenge_release\output\v5'; %output folder
output_sub = {'EDA','EMG','Resp-ECG','ViveEye','XPCAC','XPCPLT'}; %output folder names
output = cell.empty;
for i = 1:numel(output_sub)
    output = [output,{[output0 filesep output_sub{i} filesep]}];
end
output = output{dataset_idx}; %output corresponding to particular dataset of interest (defined by dataset_idx)

addpath(genpath(paceFolder))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load development data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%cd to directory of development data
cd(input_dev)
%make variable folders to keep track of file locations of interest for each
%subject
fileDirectory = dir;
dirFlags = [fileDirectory.isdir];
folders = {fileDirectory.name};
folders = folders(dirFlags);
folders = folders(3:end);

t = cell.empty; %time data
t_unique = [];
subj_list = cell.empty; %subject-run list

for i = 1:numel(folders)
    %for subject i, locate the folders that contain information of each run
    %(and generate folder3 to obtain this information); folder2 just
    %contains the dates the subjects performed the study
    cd(input_dev)
    cd(folders{i})
    fileDirectory2 = dir;
    folders2 = {fileDirectory2.name};
    cd(folders2{3})
    fileDirectory3 = dir;
    folders3 = {fileDirectory3.name};
    folders3 = folders3(3:end);
    
    for j = 1:numel(folders3)
        cd(folders3{j})
        fileDirectory4 = dir('*.mat'); %contains the names of all .mat files in a run folder
        names = {fileDirectory4.name};
        names2 = cell(size(names));
        for k = 1:numel(names)
            names_sub = strsplit(names{k},'_');
            names2{k} = names_sub{4};
        end
        
        [~,~,idx] = intersect(dataset,names2); %chooses the .mat file of interest (corresponding to the data modality of interest define by dataset_idx

        load(names{idx})
        
        %If modality is ViveEye, remove the last column
        if strcmp(dataset,'stream-lslhtcviveeye')
            data = data(:,1:end-1);
            header = header(1:end-1);
        end
        
        %if this is the first time processing the time series data (first
        %subject, first run), initialize ts_dev and ts_header (which
        %contain the time series data and corresponding names,
        %respectively)
        if i==1 && j==1
            ts_dev = cell(1,size(data,2)-1);
            ts_header = header(2:end);
            for k = 1:numel(ts_dev)
                ts_dev{k} = cell.empty;
            end
        end
        time = data(:,1); time = time-time(1); %load time and shift it relative to the first reading
        
        %reduce time to whole numbers and unique values
        time = round(time,0);
        [time,idx,~] = unique(time);
        
        t = [t,{time'}];
        t_unique = [t_unique;time];
        
        for k = 1:numel(ts_header)
            if i==1 && j==1
                ts_dev{k} = {data(idx,k+1)'};
            else
                ts_sub = ts_dev{k};
                ts_dev{k} = [ts_sub,{data(idx,k+1)'}];
            end
        end
        
        %keep track of the order of the subjects and runs as time series data is loaded
        subj_list = [subj_list; [folders{i} '-' folders3{j}]]; 
        
        cd ..
    end
    
end
cd(input_dev)
%plot original time series signals
for k = 1:numel(ts_dev)
    ctr=1;
    for i = 1:numel(folders)
        figure%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        set(gcf,'Position',[10 10 3000 4000])
        for j = 1:numel(folders3)
            subplot(3,4,j)
            plot(t{ctr},ts_dev{k}{ctr})
            title(folders3{j},'Interpreter','none')
            ctr=ctr+1;
        end
        sgtitle([ts_header{k} ', ' folders{i}],'Interpreter','none')
        saveas(gcf,[output 'raw_' ts_header{k} '_' folders{i} '.png'])
        close(gcf)
    end
end

t_unique = unique(t_unique);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load validation data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%cd to directory of validation data
cd(input_val)
%make variable folders to keep track of file locations of interest for each
%subject
fileDirectory = dir;
dirFlags = [fileDirectory.isdir];
folders = {fileDirectory.name};
folders = folders(dirFlags);
folders = folders(3:end);

t_val = cell.empty; %time data
t_val_unique = [];
subj_list_val = cell.empty; %subject-run list

for i = 1:numel(folders)
    %for subject i, locate the folders that contain information of each run
    %(and generate folder3 to obtain this information); folder2 just
    %contains the dates the subjects performed the study
    cd(input_val)
    cd(folders{i})
    fileDirectory2 = dir;
    folders2 = {fileDirectory2.name};
    cd(folders2{3})
    fileDirectory3 = dir;
    folders3 = {fileDirectory3.name};
    folders3 = folders3(3:end);
    
    for j = 1:numel(folders3)
        cd(folders3{j})
        fileDirectory4 = dir('*.mat');
        %fileDirectory4 = orderfields(fileDirectory4, [1:2,4:6,3]);
        names = {fileDirectory4.name};
        names2 = cell(size(names));
        for k = 1:numel(names)
            names_sub = strsplit(names{k},'_');
            names2{k} = names_sub{4};
        end
        
        [~,~,idx] = intersect(dataset,names2);

        load(names{idx})
        %if this is the first time processing the time series data (first
        %subject, first run), initialize ts_dev and ts_header (which
        %contain the time series data and corresponding names,
        %respectively)
        if i==1 && j==1
            ts_val = cell(1,size(data,2)-1);
            ts_header = header(2:end);
            for k = 1:numel(ts_val)
                ts_val{k} = cell.empty;
            end
        end
        time = data(:,1); time = time-time(1); %load time and shift it relative to the first reading
        
        %reduce time to whole numbers and unique values
        time = round(time,0);
        [time,idx,~] = unique(time);
        
        t_val = [t_val,{time'}];
        t_val_unique = [t_val_unique;time];
        
        for k = 1:numel(ts_header)
            if i==1 && j==1
                ts_val{k} = {data(idx,k+1)'};
            else
                ts_sub = ts_val{k};
                ts_val{k} = [ts_sub,{data(idx,k+1)'}];
            end
        end
        
        %keep track of the order of the subjects and runs as time series data is loaded
        subj_list_val = [subj_list_val; [folders{i} '-' folders3{j}]];
        
        cd ..
    end
    
end
cd(input_val)

%plot original time series signals
for k = 1:numel(ts_val)
    ctr=1;
    for i = 1:numel(folders)
        figure%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        set(gcf,'Position',[10 10 3000 4000])
        for j = 1:numel(folders3)
            subplot(3,4,j)
            plot(t_val{ctr},ts_val{k}{ctr})
            title(folders3{j},'Interpreter','none')
            ctr=ctr+1;
        end
        sgtitle([ts_header{k} ', ' folders{i}],'Interpreter','none')
        saveas(gcf,[output 'raw_val_' ts_header{k} '_' folders{i} '.png'])
        close(gcf)
    end
end

t_val_unique = unique(t_val_unique);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform FPCA (PACE) to generate features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load response metrics
cd(input_dev)
perfMetrics = readcell('PerfMetrics_test.csv');
perf_header = perfMetrics(1,:);
perf_body = cell2mat(perfMetrics(2:end,:));
diffic = perf_body(:,4); %difficulty of the task
err = perf_body(:,5:end); %pilot error
err_names = perf_header(5:end); %error names

%perform FPCA on each time series in our dataset
%dev set
fve = cell(size(ts_dev)); %fraction of variance explained (fve)
fve2 = cell(size(ts_dev)); %second fve variable made for formatting reasons
fpcs = cell(size(ts_dev)); %fpcs variables stores the time series values of the resulting eigenfunctions
score = cell(size(ts_dev)); %score contains the score features that are used for input to the ML algorithm
mu = cell(size(ts_dev)); %mu contains the smoothed mean curve that is calculated from the pooled time series data
t_fpc = cell(size(ts_dev)); %t_fpc contains the time vectore corresponding to fpcs and mu
%val set
score_val = cell(size(ts_val));

for i = 1:numel(ts_dev)
    %obtain some first order statistics (mean, sd, skew, kurtosis)
    ts_dev_mean = [];
    ts_dev_sd = [];
    ts_dev_skew = [];
    ts_dev_kurt = [];
    
    ts_val_mean = [];
    ts_val_sd = [];
    ts_val_skew = [];
    ts_val_kurt = [];
    
    for j = 1:numel(ts_dev{i})
        ts_dev_mean = [ts_dev_mean; mean(ts_dev{i}{j})];
        ts_dev_sd = [ts_dev_sd; std(ts_dev{i}{j})];
        ts_dev_skew = [ts_dev_skew; skewness(ts_dev{i}{j})];
        ts_dev_kurt = [ts_dev_kurt; kurtosis(ts_dev{i}{j})];
    end
    for j = 1:numel(ts_val{i})
        ts_val_mean = [ts_val_mean; mean(ts_val{i}{j})];
        ts_val_sd = [ts_val_sd; std(ts_val{i}{j})];
        ts_val_skew = [ts_val_skew; skewness(ts_val{i}{j})];
        ts_val_kurt = [ts_val_kurt; kurtosis(ts_val{i}{j})];
    end
    
    ts_dev_1storder = [ts_dev_mean,ts_dev_sd,ts_dev_skew,ts_dev_kurt]; ts_dev_1storder = num2cell(ts_dev_1storder);
    ts_val_1storder = [ts_val_mean,ts_val_sd,ts_val_skew,ts_val_kurt]; ts_val_1storder = num2cell(ts_val_1storder);
    
    header_1storder= {'Mean','SD','Skew','Kurtosis'};
    for j = 1:numel(header_1storder)
       header_1storder{j} = [ts_header{i} '-' header_1storder{j}]; 
    end
        
    %perform FPCA
    p = setOptions('yname',ts_header{i},'selection_k','FVE','FVE_threshold',0.95,'screePlot',1,'designPlot',1,'corrPlot',1,'verbose','on');
    X = FPCA(ts_dev{i},t,p);
    fve0 = getVal(X,'FVE');
    fve0_sub = [0,fve0(1:end-1)];
    fve{i} = round(fve0-fve0_sub,3);
    fve2{i} = fve0-fve0_sub;
    fpcs{i} = getVal(X,'phi');
    score{i} = getVal(X,'xi_est');
    mu{i} = getVal(X,'mu');
    t_fpc{i} = getVal(X,'out1');
    
    %obtain FPCs associated with validation set
    [~, score_val{i}, ~] =  FPCApred(X, ts_val{i}, t_val);
    
    figure%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    set(gcf,'Position',[10 10 2000 1000*size(score{i},2)])
    
    %make some visuals and check discrimination capabilities of PCs
    %plot FPC curves
    %then below each curve, plot box plots of scores based on each difficulty level
    %(maybe also indicate if any significant differences between each
    %group-2 sample t-test)
    for j = 1:size(score{i},2)
        subplot(2,size(score{i},2),j)
        yline(0)
        hold on
        plot(t_fpc{i},fpcs{i}(:,j),'red')
        xlabel('Time')
        ylabel(['FPC ' num2str(j)])
        title(['FVE: ' num2str(fve{i})])
        
        subplot(2,size(score{i},2),size(score{i},2)+j)
        boxplot(score{i}(:,j),diffic)
        ylabel(['FPC ' num2str(j)])
        xlabel('Difficulty')
    end
    
    sgtitle(ts_header{i},'Interpreter','none')
    saveas(gcf,[output 'FPCs_' ts_header{i} '.png'])
    close(gcf)
    
    figure%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    set(gcf,'Position',[10 10 4000 1000*size(score{i},2)])
    
    %then do a simple linear plot with PC and the error measurements
    %(indicate if any correlations are significant; maybe also indicate
    %which difficulty level each instances are)
    for j = 1:size(score{i},2)
        err_no = 1;
        subplot(4,size(score{i},2),j)
        [r,p] = corrcoef(score{i}(:,j),err(:,err_no));
        r = round(r(1,2),3); p = round(p(1,2),5);
        plot(score{i}(diffic==1,j),err(diffic==1,err_no),'.','color','cyan','MarkerSize',20)
        hold on
        plot(score{i}(diffic==2,j),err(diffic==2,err_no),'.','color','green','MarkerSize',20)
        hold on
        plot(score{i}(diffic==3,j),err(diffic==3,err_no),'.','color','magenta','MarkerSize',20)
        hold on
        plot(score{i}(diffic==4,j),err(diffic==4,err_no),'.','color','red','MarkerSize',20)
        legend('1','2','3','4')
        xlabel(['FPC ' num2str(j)])
        ylabel(err_names{err_no},'Interpreter','none')
        title(['R = ' num2str(r) ' (p = ' num2str(p) ')'])
        err_no = err_no+1;
        
        subplot(4,size(score{i},2),size(score{i},2)+j)
        [r,p] = corrcoef(score{i}(:,j),err(:,err_no));
        r = round(r(1,2),3); p = round(p(1,2),5);
        plot(score{i}(diffic==1,j),err(diffic==1,err_no),'.','color','cyan','MarkerSize',20)
        hold on
        plot(score{i}(diffic==2,j),err(diffic==2,err_no),'.','color','green','MarkerSize',20)
        hold on
        plot(score{i}(diffic==3,j),err(diffic==3,err_no),'.','color','magenta','MarkerSize',20)
        hold on
        plot(score{i}(diffic==4,j),err(diffic==4,err_no),'.','color','red','MarkerSize',20)
        legend('1','2','3','4')
        xlabel(['FPC ' num2str(j)])
        ylabel(err_names{err_no},'Interpreter','none')
        title(['R = ' num2str(r) ' (p = ' num2str(p) ')'])
        err_no = err_no+1;
        
        subplot(4,size(score{i},2),size(score{i},2)*2+j)
        [r,p] = corrcoef(score{i}(:,j),err(:,err_no));
        r = round(r(1,2),3); p = round(p(1,2),5);
        plot(score{i}(diffic==1,j),err(diffic==1,err_no),'.','color','cyan','MarkerSize',20)
        hold on
        plot(score{i}(diffic==2,j),err(diffic==2,err_no),'.','color','green','MarkerSize',20)
        hold on
        plot(score{i}(diffic==3,j),err(diffic==3,err_no),'.','color','magenta','MarkerSize',20)
        hold on
        plot(score{i}(diffic==4,j),err(diffic==4,err_no),'.','color','red','MarkerSize',20)
        legend('1','2','3','4')
        xlabel(['FPC ' num2str(j)])
        ylabel(err_names{err_no},'Interpreter','none')
        title(['R = ' num2str(r) ' (p = ' num2str(p) ')'])
        err_no = err_no+1;
        
        subplot(4,size(score{i},2),size(score{i},2)*3+j)
        [r,p] = corrcoef(score{i}(:,j),err(:,err_no));
        r = round(r(1,2),3); p = round(p(1,2),5);
        plot(score{i}(diffic==1,j),err(diffic==1,err_no),'.','color','cyan','MarkerSize',20)
        hold on
        plot(score{i}(diffic==2,j),err(diffic==2,err_no),'.','color','green','MarkerSize',20)
        hold on
        plot(score{i}(diffic==3,j),err(diffic==3,err_no),'.','color','magenta','MarkerSize',20)
        hold on
        plot(score{i}(diffic==4,j),err(diffic==4,err_no),'.','color','red','MarkerSize',20)
        legend('1','2','3','4')
        xlabel(['FPC ' num2str(j)])
        ylabel(err_names{err_no},'Interpreter','none')
        title(['R = ' num2str(r) ' (p = ' num2str(p) ')'])
    end
    
    sgtitle(ts_header{i},'Interpreter','none')
    saveas(gcf,[output 'FPCsErrCorr_' ts_header{i} '.png'])
    close(gcf)
    
    %write key files to csv
    %score for development set
    score_body = num2cell(score{i}); score_body_size = size(score_body,2); score_body = [ts_dev_1storder,score_body];
    score_body = [subj_list,score_body];
    score_header = [{'Subject-Run'},header_1storder];
    for j = 1:score_body_size
        score_header = [score_header,{[ts_header{i} '-FPC' num2str(j)]}];
    end
    score_file = [score_header;score_body];
    writecell(score_file,[output 'score_' ts_header{i} '.csv'])
    %score for validation set
    score_val_body = num2cell(score_val{i}); score_val_body_size = size(score_val_body,2); score_val_body = [ts_val_1storder,score_val_body];
    score_val_body = [subj_list_val,score_val_body];
    score_val_header = [{'Subject-Run'},header_1storder];
    for j = 1:score_val_body_size
        score_val_header = [score_val_header,{[ts_header{i} '-FPC' num2str(j)]}];
    end
    score_val_file = [score_val_header;score_val_body];
    writecell(score_val_file,[output 'score_val_' ts_header{i} '.csv'])
    
    %FVE (Fraction of Variance Explained)
    fve_header = score_header(6:end);
    fve_body = num2cell(fve2{i}); fve_body = fve_body(1:score_body_size);
    fve_file = [fve_header;fve_body];
    writecell(fve_file,[output 'fve_' ts_header{i} '.csv'])
    
    %Time series values of FPC functions (eigenvectors as a function of
    %time)
    fpc_header = fve_header; fpc_header = [{'Time','Mean Curve'},fpc_header];
    t_fpc_col = t_fpc{i}'; mu_col = mu{i}';
    fpc_body = [t_fpc_col,mu_col,fpcs{i}]; fpc_body = num2cell(fpc_body);
    fpc_file = [fpc_header;fpc_body];
    writecell(fpc_file,[output 'fpcs_' ts_header{i} '.csv']);
end
