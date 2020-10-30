clear 
load iris.dat

% Πρώτα 25 στοιχεία κάθε κλάσης
training_set = iris(1:25,:)
training_set = cat(1,iris(51:75,:),training_set)
training_set = cat(1,iris(101:125,:),training_set)

testing_set = iris(26:50,:)
testing_set = cat(1,iris(76:100,:),testing_set)
testing_set = cat(1,iris(126:150,:),testing_set)

training_data = training_set(:,1:4)
training_targets = training_set(:,5)
testing_data = testing_set(:,1:4)
testing_targets = testing_set(:,5)

fis = fcm_fis(training_data,training_targets)
[trainFis, trainError, stepSize, checkFis, checkError] = ...
    anfis(training_set,fis,100,  [], testing_set);

output = round(evalfis(trainFis,testing_data));

get_success_results(output,testing_targets)
[success_rate,successful] = get_success_results(output,testing_targets)

plotClassification(output,testing_targets)

showrule(fis)

f = figure(2); %to show the 2 figures into different windows
plotInputs(fis)

%------ Functions ------

%fuzzy c-means
function fis = fcm_fis(training_data,training_targets)
opts = genfisOptions('FCMClustering');
opts.NumClusters = 3;
fis = genfis(training_data,training_targets,opts);
end 


%gridpartition
function fis = gridpartition_fis(training_data,training_targets)
opts = genfisOptions('GridPartition');
opts.NumMembershipFunctions = 3;
fis = genfis(training_data,training_targets,opts);
end


%substractive clustering
function fis = substractive_fis(training_data,training_targets)
opts= genfisOptions('SubtractiveClustering');
fis= genfis(training_data,training_targets,opts);
end

%run algorithm
function trainfis = run_algorithm(training_set,algo,epochs,testing_set)
[trainFis, trainError, stepSize, checkFis, checkError] = ...
    anfis(training_set, algo,100,  [], testing_set);
end

%plot inputs
function plotInputs(fis)
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
function [success_rate,successful] = get_success_results(output,testing_targets)
successful = size(find((output == testing_targets) == 0), 1);
success_rate = round((successful/75)*100)
end

%plot classification
function plotClassification(output,testing_targets)
plot(output)
hold on
plot(testing_targets,'o')
hold off
end