% ------------------------------------------------------------------------
% Script CreateDataset.m
% ------------------------------------------------------------------------
% 
%
% ------------------------------------------------------------------------
% In order to execute this script, the following functions are necessary: 
% 
% ------------------------------------------------------------------------

%% Initial Parameters

clear all
%sys_slash='/'; % for mac OS uncoment this variable
sys_slash='\'; % for Windows uncoment this variable

training_devices = {'S20', 'S7', 'POCO', 'tabS7', 'A12'}; 
testing_devices = {'S20', 'S7', 'POCO', 'tabS7', 'A12'};

data_dir = 'FORMATTED'; % path were the dataset is saved

%% Start making the formatted dataset
% LOAD ALL DATA

res_train = [];
res_test = [];

for device_n = 1:size(training_devices,2) 
    
    selected_device = training_devices{device_n};
    filename = fullfile(data_dir,['Formatted_data_',selected_device,'.mat']);
    
    load(filename)
    res_train = [res_train, results_train];
    %res_test = [res_test, results_test];
    
end
for device_n = 1:size(testing_devices,2) 
    
    selected_device = testing_devices{device_n};
    filename = fullfile(data_dir,['Formatted_data_',selected_device,'.mat']);
    
    load(filename)
    %res_train = [res_train, results_train];
    res_test = [res_test, results_test];
    
end
% GET ALL UNIQUE APs
wifis_all = [];
nsamples_wifi_train = 0;
nsamples_wifi_test = 0;
locations_all_test = [];
for device_n = 1:size(training_devices,2) 
    for fp = 1:size(res_train(device_n).Wifi,2)
        wifis_all = [wifis_all; res_train(device_n).Wifi(fp).measured(:,1)];
    end
end
for device_n = 1:size(testing_devices,2) 
    for fp = 1:size(res_test(device_n).Wifi,2)
        wifis_all = [wifis_all; res_test(device_n).Wifi(fp).measured(:,1)];
        locations_all_test = [locations_all_test; res_test(device_n).Wifi(fp).coords];
    end
end

rng(2)
locations_all_test = unique(locations_all_test,'rows');

percentage_to_test = 0.50;
random_indices = randperm(size(locations_all_test,1));
indexes_to_test = random_indices < size(locations_all_test,1)*percentage_to_test;
indexes_to_train = random_indices >= size(locations_all_test,1)*percentage_to_test;

location_to_test = locations_all_test(indexes_to_test,:);
locations_to_train = locations_all_test(indexes_to_train,:);

for device_n = 1:size(training_devices,2) 
    nsamples_wifi_train = nsamples_wifi_train + size(res_train(device_n).Wifi,2);
end
for device_n = 1:size(training_devices,2)
    nsamples_wifi_test = nsamples_wifi_test + size(res_test(device_n).Wifi,2);
end

% FIND UNIQUE MACS
wifis_unique = unique(wifis_all);

%% MAKE DATASET
% Initialize matrices

database.trainingMacs = ones(nsamples_wifi_train, size(wifis_unique,1)).*NaN;
database.testMacs = ones(10, size(wifis_unique,1)).*NaN;
database.trainingLabels = zeros(nsamples_wifi_train, 6);
database.testLabels = zeros(10, 6);

% fill the matrices with the adequate data
% 1. training data
iter_train = 1;
iter_test = 1;

num_train = zeros(1,5);
num_test = zeros(1,5);

for device_n = 1:size(training_devices,2) 
    for fp = 1:size(res_train(device_n).Wifi,2)
        
        fingerprint_rss = res_train(device_n).Wifi(fp).measured;
        fingerprint_xy = res_train(device_n).Wifi(fp).coords;
        
        for nmac = 1:size(fingerprint_rss,1)
           mac = fingerprint_rss(nmac,1);
           database.trainingMacs(iter_train, wifis_unique == mac) = fingerprint_rss(nmac,2);
        end
        database.trainingLabels(iter_train,1:2) = fingerprint_xy;
        database.trainingLabels(iter_train,6) = device_n;
        iter_train = iter_train +1;
        num_train(device_n) = num_train(device_n)+1;
    end
end

for device_n = 1:size(testing_devices,2) 
    for fptr = 1:size(res_test(device_n).Wifi,2)
        
        fingerprint_rss = res_test(device_n).Wifi(fptr).measured;
        fingerprint_xy = res_test(device_n).Wifi(fptr).coords;
        
        if ismember(fingerprint_xy, location_to_test, 'rows')
            for nmac = 1:size(fingerprint_rss,1)
               mac = fingerprint_rss(nmac,1);
               database.testMacs(iter_test, wifis_unique == mac) = fingerprint_rss(nmac,2);
            end
            database.testLabels(iter_test,1:2) = fingerprint_xy;
            database.testLabels(iter_test,6) = device_n;
            iter_test = iter_test +1;
            num_test(device_n) = num_test(device_n)+1;
        elseif ismember(fingerprint_xy, locations_to_train, 'rows')
            for nmac = 1:size(fingerprint_rss,1)
               mac = fingerprint_rss(nmac,1);
               database.trainingMacs(iter_train, wifis_unique == mac) = fingerprint_rss(nmac,2);
            end
            database.trainingLabels(iter_train,1:2) = fingerprint_xy;
            database.trainingLabels(iter_train,6) = device_n;
            iter_train = iter_train +1; 
            num_train(device_n) = num_train(device_n)+1;
        end
    end
end

database.trainingMacs(isnan(database.trainingMacs)) = 100;
database.trainingMacs(database.trainingMacs == 0) = 100;
database.testMacs(isnan(database.testMacs)) = 100;
database.testMacs(database.testMacs == 0) = 100;

database.testMacs();

save('DATASET/TUJI1.mat','database')

temp = database.trainingMacs;
writematrix(temp, 'DATASET/RSS_training.csv')
temp = database.testMacs;
writematrix(temp, 'DATASET/RSS_testing.csv')
temp = database.trainingLabels;
writematrix(temp, 'DATASET/Coordinates_training.csv')
temp = database.testLabels;
writematrix(temp, 'DATASET/Coordinates_testing.csv')

device_labels = table(training_devices',[1:5]','VariableNames',{'Device', 'Label'});
% save('DATASET/Device_labels.mat','device_labels')
writetable(device_labels, 'DATASET/Device_labels.csv')












