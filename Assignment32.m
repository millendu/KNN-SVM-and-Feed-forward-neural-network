%%%%% loading test data below %%%%%
load X_test.mat
load X_train.mat
load y_test.mat
load y_train.mat

%%%% transposing y_train data that is read %%%%
y_trans = transpose(y_train);

%%%%%% code to generate and train a model based on knn classifier %%%%%%
mdl = fitcknn(X_train,y_trans,'NumNeighbors',5);
predict_knn = predict(mdl,X_test);
count_knn=0;

for i=1: numel(predict_knn)
    if predict_knn(i,1) == y_test(1,i)
        count_knn = count_knn + 1;
    end
end

%%% calculating accuracy of the model generated above %%%%
accuracy_knn = count_knn*100/numel(predict_knn);



%%%% code to generate and train a model based on svm classifier %%%%
Mdl = fitcecoc(X_train,y_train.','Learners',templateSVM('KernelFunction','polynomial','PolynomialOrder',2));

predict_svm = predict(Mdl,X_test(:,:));
output=y_test(:,:).';
count_svm = 0;
for i = 1:numel(output)
    if (output(i,1) == predict_svm(i,1))
        count_svm = count_svm + 1;
    end
end

%%% calculating accuracy of the model generated above %%%%
accuracy_svm = (count_svm/numel(output)) * 100;


%%%% code to train a feedforward nueral network with 25 nuerons %%%%%
ffn = feedforwardnet(25);
x_trans = transpose(X_train);
y_vec = full(ind2vec(y_train));
ffn = train(ffn,x_trans,y_vec);    %%%% training the feed forward  network with training dataset %%%%
view(ffn)   
x_trans1 = transpose(X_test);
y_new = ffn(x_trans1);             %%%% validating the generated neural network with test data %%%%
final_result = vec2ind(y_new);
count = 0;
for i = 1:numel(final_result)
    if final_result(i) == y_test(i)
        count = count +1;
    end
end

%%% calculating accuracy of the model generated above %%%%
accuracy_fnn = count/numel(final_result) *100;


%%% displaying the calculated accuracies %%%
result_knn = ['The accuracy_knn =', num2str(accuracy_knn)];
result_ffn = ['The accuracy_ffn =', num2str(accuracy_fnn)];
result_svm = ['The accuracy_svm =', num2str(accuracy_svm)];



disp(result_knn);
disp(result_ffn);
disp(result_svm);