%%%%% loading test data below %%%%%
load X_test.txt
load X_train.txt
load y_test.txt
load y_train.txt

%%%% transposing y_train data that is read %%%%
y_trans = transpose(y_train);

%%%%%% code to generate and train a model based on knn classifier %%%%%%
mdl = fitcknn(X_train,y_trans,'NumNeighbors',5);
predict_knn = predict(mdl,X_test);
count_knn=0;

for i=1: numel(predict_knn)
    if predict_knn(i,1) == y_test(i,1)
        count_knn = count_knn + 1;
    end
end

%%% calculating accuracy of the model generated above %%%%
accuracy_knn = count_knn*100/numel(predict_knn)



%%%% code to generate and train a model based on svm classifier %%%%
Mdl = fitcecoc(X_train,y_train.','Learners',templateSVM('KernelFunction','polynomial','PolynomialOrder',2));

predict_svm = predict(Mdl,X_test(:,:));
p = transpose(predict_svm)
output=y_test(:,:).';
count_svm = 0;
for i = 1:numel(output)
    if (output(1,i) == p(1,i))
        count_svm = count_svm + 1;
    end
end

%%% calculating accuracy of the model generated above %%%
accuracy_svm = (count_svm/numel(output)) * 100;

result_knn = ['The accuracy_knn =', num2str(accuracy_knn)];
result_svm = ['The accuracy_svm =', num2str(accuracy_svm)];

%%% displaying the calculated accuracies %%%
disp(result_knn);
disp(result_svm);
