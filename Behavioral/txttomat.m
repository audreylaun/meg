%% turning a text file into a mat file %%

%LOAD THE FILE 
filename = '/Users/admin/Desktop/interwoven_counterbalancing_D.txt';
data = importdata(filename); %loads a struct with numeric and text data separated 

% Gather numeric values 
wordTriggers= data.data(:,1); %Trigger for when the image is shown
triggerList = data.data(:,2); %Trigger for audio onset
offsetTriggers = data.data(:,3); %Trigger for audio offset 
condition = data.data(:,4); %Condition (prod or comp) 

% Gather string values 
wordList = data.textdata(:,2); %Names of photos 
wavList = data.textdata(:,1); %Names of audios 
%compList = data.textdata(:,3); %words for comprehension questions 

clear data; 
clear filename; 

%%SAVE THE WORKSPACE TO GET THE .MAT (Home tab) 