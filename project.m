clear 
load iris.dat

% Πρώτα 25 στοιχεία κάθε κλάσης
global training_set
training_set = iris(1:25,:);
training_set = cat(1,iris(51:75,:),training_set);
training_set = cat(1,iris(101:125,:),training_set);

global testing_set
testing_set = iris(26:50,:);
testing_set = cat(1,iris(76:100,:),testing_set);
testing_set = cat(1,iris(126:150,:),testing_set);

global training_data;
global training_targets;
global testing_data;
global testing_targets;
training_data = training_set(:,1:4);
training_targets = training_set(:,5);
testing_data = testing_set(:,1:4);
testing_targets = testing_set(:,5);

global figure_index;
figure_index=0;
fcm();
substractive();
gridpartition();
%------ Functions ------


%evalute algorithm
function evaluate_algorithm(fis)
global testing_data;
global testing_targets;
global figure_index;
trainFis = train_algorithm(fis,100);
output = round(evalfis(trainFis,testing_data));
[success_rate,successful] = get_success_results(output);
plotClassification(output);
plotInputs(fis);
end

%train algorithm
function trainfis = train_algorithm(fis,epochs)
global testing_set;
global training_set;
[trainfis, trainError, stepSize, checkFis, checkError] = ...
    anfis(training_set, fis,epochs,  [], testing_set);
end


%plot inputs
function plotInputs(fis)
global figure_index;
figure_index = figure_index+1;
figure(figure_index);
[x,mf] = plotmf(fis,'input',1);
subplot(2,2,1)
plot(x,mf)
xlabel('Μήκος Σεπάλων')
[x,mf] = plotmf(fis,'input',2);
subplot(2,2,2)
plot(x,mf)
xlabel('Πλάτος Σεπάλων')
[x,mf] = plotmf(fis,'input',3);
subplot(2,2,3)
plot(x,mf)
xlabel('Μήκος Πετάλων')
[x,mf] = plotmf(fis,'input',4);
subplot(2,2,4)
plot(x,mf)
xlabel('Πλάτος Πετάλων')
end

%success rate
function [success_rate,successful] = get_success_results(output)
global testing_targets;
successful = size(find((output == testing_targets) == 0), 1);
success_rate = round((successful/75)*100);
end

%plot classification
function plotClassification(output)
global figure_index;
figure_index = figure_index+1;
figure(figure_index);
global testing_targets;
plot(output)
hold on
plot(testing_targets,'o')
hold off
end

%fuzzy c-means
function fis = fcm_fis()
global training_data;
global training_targets;
opts = genfisOptions('FCMClustering');
opts.NumClusters = 3;
fis = genfis(training_data,training_targets,opts);
end 

%fuzzy c-means RUN
function [success_rate,successful] = fcm()
fis = fcm_fis();
evaluate_algorithm(fis);
end


%gridpartition
function fis = gridpartition_fis()
global training_data;
global training_targets;
opts = genfisOptions('GridPartition');
opts.NumMembershipFunctions = 3;
fis = genfis(training_data,training_targets,opts);
end

%gridpartition RUN
function [success_rate,successful] = gridpartition(figure_index)
fis = gridpartition_fis();
evaluate_algorithm(fis);
end

%substractive clustering
function fis = substractive_fis()
global training_data;
global training_targets;
opts= genfisOptions('SubtractiveClustering');
fis= genfis(training_data,training_targets,opts);
end

%substractive clustering RUN
function [success_rate,successful] = substractive(figure_index)
fis = substractive_fis();
evaluate_algorithm(fis);
end

