clear 
load iris.dat
RADIUS=0.8;
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

%fuzzy c-means
opt2 = genfisOptions('FCMClustering');
opt2.NumClusters = 3;
fis2= genfis(training_data,training_targets,opt2);


%substractive clustering
fis = genfis2(training_data,training_targets,RADIUS,[min(iris); max(iris)]);

%gridpartition
opt2 = genfisOptions('GridPartition');
opt2.NumMembershipFunctions = 3;
fis3 = genfis(training_data,training_targets,opt2);


 [trainFis, trainError, stepSize, checkFis, checkError] = ...
    anfis(training_set, fis,100,  [], testing_set);



trainFisOut = round(evalfis(trainFis,testing_data));

failed_tests = size(find((trainFisOut == testing_targets) == 0), 1);
fail_rate_percentage = round((failed_tests/75)*100)
test = [trainFisOut(:),testing_targets(:),trainFisOut(:)~=testing_targets(:)]

plot(trainFisOut)
hold on
plot(testing_targets,'o')
hold off

showrule(fis)

[x,mf] = plotmf(fis,'input',1);
subplot(2,2,1)
plot(x,mf)
xlabel('Membership Functions for Input 1')
[x,mf] = plotmf(fis,'input',2);
subplot(2,2,2)
plot(x,mf)
xlabel('Membership Functions for Input 2')
[x,mf] = plotmf(fis,'input',3);
subplot(2,2,3)
plot(x,mf)
xlabel('Membership Functions for Input 3')
[x,mf] = plotmf(fis,'input',4);
subplot(2,2,4)
plot(x,mf)
xlabel('Membership Functions for Input 4')
