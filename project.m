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

% global variables, χρήση στις συναρτήσεις
global training_data;
global training_targets;
global testing_data;
global testing_targets;
training_data = training_set(:,1:4);
training_targets = training_set(:,5);
testing_data = testing_set(:,1:4);
testing_targets = testing_set(:,5);


% Αρχικοποίηση Cell για την αποθήκευση αριθμητικών αποτελεσμάτων
global results;
results = {};

% Κλήση αλγορίθμων
fcm();
gridpartition();
substractive();

% Εμφάνιση αριθμητικών αποτελεσμάτων για σύγκριση
results

%------ Functions ------

%evalute algorithm
function evaluate_algorithm(fis,algo_title)
global results;
global testing_data;
global testing_targets;

% Εκπαίδευση 
trainFis = train_algorithm(fis,100);

% Έξοδος αποτελεσμάτων νευρώνα στα testing δεδομένα
output = round(evalfis(trainFis,testing_data));

[successful,success_rate] =get_success_results(output);

% format 
success_results = {algo_title,successful,success_rate};

% Αποθήκευση αριθμητικών αποτελεσμάτων
results = [results;success_results];

% Εμφάνιση γραφημάτων
plotClassification(output,algo_title,successful);
plotInputs(fis,algo_title);
end

%train algorithm
function trainfis = train_algorithm(fis,epochs)
global testing_set;
global training_set;

% Εκπαίδευση Νευρώνα
trainfis  = ...
    anfis(training_set, fis,epochs,  [], testing_set);
end


%plot inputs
function plotInputs(fis,algo_title)
figure('Name',strcat('Membership Function graphs using ',algo_title));
[x,mf] = plotmf(fis,'input',1);
subplot(2,2,1);
plot(x,mf);
xlabel('Μήκος Σεπάλων');
[x,mf] = plotmf(fis,'input',2);
subplot(2,2,2);
plot(x,mf);
xlabel('Πλάτος Σεπάλων');
[x,mf] = plotmf(fis,'input',3);
subplot(2,2,3);
plot(x,mf);
xlabel('Μήκος Πετάλων');
[x,mf] = plotmf(fis,'input',4);
subplot(2,2,4);
plot(x,mf);
xlabel('Πλάτος Πετάλων');
end

%success rate
function [successful,success_rate] = get_success_results(output)
    global testing_targets;

    % Εύρεση επιτυχημένων
    successful = size(find((output == testing_targets) == 1), 1);

    % Ποσοστό επιτυχημένων
    success_rate = sprintf("%d %%",round((successful/75)*100));
end

%plot classification
function plotClassification(output,algo_title,successful)
    figure('Name','Classification success plot');
    global testing_targets;
    plot(output,'x');
    hold on
    plot(testing_targets,'o');
    hold off
    title(strcat(algo_title,sprintf(' successful classifications: %d out of 75',successful)));
end

%fuzzy c-means
function fis = fcm_fis()
    global training_data;
    global training_targets;
    opts = genfisOptions('FCMClustering');

    % Clusters όσοι και τύποι λουλουδιών
    opts.NumClusters = 3;

    fis = genfis(training_data,training_targets,opts);
    end 

    %fuzzy c-means RUN
    function fcm()
    fis = fcm_fis();
    evaluate_algorithm(fis,'FCM');
end


%gridpartition
function fis = gridpartition_fis()
    global training_data;
    global training_targets;
    opts = genfisOptions('GridPartition');

    % Συναρτήσεις συμμετοχής όσες και τύποι λουλουδιών
    opts.NumMembershipFunctions = 3;
    fis = genfis(training_data,training_targets,opts);
end

%gridpartition RUN
function gridpartition()
    fis = gridpartition_fis();
    evaluate_algorithm(fis,'gridPartition');
end

%substractive clustering
function fis = substractive_fis()
    global training_data;
    global training_targets;
    opts = genfisOptions('SubtractiveClustering');
    fis= genfis(training_data,training_targets,opts);
end

%substractive clustering RUN
function substractive()
    fis = substractive_fis();
    evaluate_algorithm(fis,'substractiveClustering');
end

