
% get_adult();
load train_data.mat
load train_label.mat
load test_data.mat
load test_label.mat

% Make data label -1 and 1
n= length(train_label);
for i = 1:n
    if(train_label(i)== 0)
        train_label(i) = -1;
    end
end

% Make data label -1 and 1
n= length(test_label);
for i = 1:n
    if(test_label(i)== 0)
        test_label(i) = -1;
    end
end

% Perofrmance of single learner
% 1. GDA
% 2. knn (NumNeighbors = 30)
% 3. Naive Bayes
% 4. Logistic Regression
% 5. SVM (rbf)
weak_learner=fitcdiscr(train_data,train_label);
predicted=predict(weak_learner, train_data);
CCR3_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR3 = 1- sum(predicted ~= test_label)/length(predicted);

weak_learner=fitcknn(train_data,train_label,'NumNeighbors',30);
predicted=predict(weak_learner, train_data);
CCR4_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR4 = 1- sum(predicted ~= test_label)/length(predicted);

weak_learner=fitcnb(train_data,train_label);
predicted=predict(weak_learner, train_data);
CCR5_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR5 = 1- sum(predicted ~= test_label)/length(predicted);

weak_learner=fitclinear(train_data,train_label,'Learner','logistic');
predicted=predict(weak_learner, train_data);
CCR6_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR6 = 1- sum(predicted ~= test_label)/length(predicted);

weak_learner=fitcsvm(train_data,train_label,'KernelFunction','rbf');
predicted=predict(weak_learner, train_data);
CCR7_1 = 1- sum(predicted ~= train_label)/length(predicted);
predicted=predict(weak_learner, test_data);
CCR7 = 1- sum(predicted ~= test_label)/length(predicted);

% Choosen Weak classifiers as stacking method:
% 1. GDA
% 2. knn (NumNeighbors = 30)
% 3. Naive Bayes
% 4. Logistic Regression
% 5. SVM (rbf)

Xtrain=train_data;
Ytrain =train_label;
Xtest = test_data;
Ytest = test_label;



Classifiers=5;




for T=1:Classifiers


        if(T== 1)
        %gda
        gda=fitcdiscr(Xtrain,Ytrain);   
        train_predict(:,T) = predict(gda, train_data);
        test_predict(:,T)  = predict(gda, test_data);
        end
    
        if(T == 2)
        %knn
        knn=fitcknn(Xtrain,Ytrain,'NumNeighbors',30);
        train_predict(:,T) = predict(knn, train_data);
        test_predict(:,T)  = predict(knn, test_data);
        end

        if(T ==3)
        %NB
        nb=fitcnb(Xtrain,Ytrain);
        train_predict(:,T) = predict(nb, train_data);
        test_predict(:,T)  = predict(nb, test_data);
        end

    
        if(T ==4)
        %logistic regression
        regression=fitclinear(Xtrain,Ytrain,'Learner','logistic');
        train_predict(:,T) = predict(regression, train_data);
        test_predict(:,T)  = predict(regression, test_data);
        end

     
        if(T ==5)
        %svm
        svm=fitcsvm(Xtrain,Ytrain,'KernelFunction','rbf');
        train_predict(:,T) = predict(svm, train_data);
        test_predict(:,T)  = predict(svm, test_data);
        end

     
    % final vote
    train_ada_QDA(:,T)=mode(train_predict,2);
    train_ada_CCR(T) = 1- sum(train_ada_QDA(:,T) ~= train_label) / length(train_label);
    % for test set
    test_ada_QDA(:,T)=mode(test_predict,2);
    test_ada_CCR(T) = 1- sum(test_ada_QDA(:,T) ~= test_label) / length(test_label);
    
    
end




