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

fis = genfis(training_data,training_targets);

[trainFis, trainError, stepSize, checkFis, checkError] = ...
    anfis(training_set, fis, 8000, [], testing_set);


trainFisOut = round(evalfis(testing_data, trainFis));

badCheckFis = size(find((trainFisOut == testing_targets) == 0), 1);

test = [trainFisOut(:),testing_targets(:)]