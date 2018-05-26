% TP pitch and temporal scale modifications

% load aeiou;
Fs = 22050;

[s,Fs] = audioread('aeiou.wav');
s = s';

soundsc(s,Fs);
fprintf('Reading Original sound. Press any key to continue...\n');
pause

A = AnalysisPitchMarks(s,Fs);

n_marks = size(A,2);
B = zeros(2,n_marks);

B(1,:) = A(1,:);
B(2,:) = 1:1:n_marks;

y = Synthesis(s,Fs,A,B);
soundsc(y,Fs);
audiowrite('aeiou_synthesis.wav',y,Fs);
fprintf('Reading re-synthesized sound. Press any key to continue...\n');
pause

alpha = 2;
B = ChangeTimeScale(alpha,A,Fs);

y_alpha = Synthesis(s,Fs,A,B);
soundsc(y_alpha,Fs);
audiowrite('aeiou_timechange.wav',y_alpha,Fs);
fprintf('Reading time scale changed sound. Press any key to continue...\n');
pause

beta = 2;
B = ChangePitchScale(beta,A,Fs);

y_beta = Synthesis(s,Fs,A,B);
soundsc(y_beta,Fs);
audiowrite('aeiou_pitchchange.wav',y_beta,Fs);
fprintf('Reading pitch changed sound. Press any key to continue...\n');
pause


B = ChangeBothScales(alpha,beta,A,Fs);

y_both = Synthesis(s,Fs,A,B);
soundsc(y_both,Fs);
audiowrite('aeiou_bothchange.wav',y_beta,Fs);
fprintf('Reading both scales changed sound. Press any key to continue...\n');
pause





