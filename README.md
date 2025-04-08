# BehaveMine
The data and source code of BehaveMine

\begin{table*} 
    \setlength\tabcolsep{1.2pt} 
    \centering
    \begin{tabular}{cccccccccccccccccc}
        \hlineB{2}
        \multirow{2}{*}{Model}&&\multicolumn{4}{c}{HH101}&\hspace{0.1cm}&\multicolumn{4}{c}{HH103}&\hspace{0.1cm}&\multicolumn{4}{c}{HH104}\\
        \cline{3-6}\cline{8-11}\cline{13-16}
        &&\hspace{0.1cm}Top-1&\hspace{0.1cm}Top-3&\hspace{0.1cm}Top-5&\hspace{0.05cm}Micro-F1&\hspace{0.1cm}&\hspace{0.1cm}Top-1&\hspace{0.1cm}Top-3&\hspace{0.1cm}TOP5&\hspace{0.05cm}Micro-F1&\hspace{0.1cm}&\hspace{0.1cm}Top-1&\hspace{0.1cm}Top-3&\hspace{0.1cm}Top-5&\hspace{0.05cm}Micro-F1\\
        \hline
        Decoder 
        &&\hspace{0.1cm}0.457&\hspace{0.1cm}0.652&\hspace{0.1cm}0.748&{0.342}   
        &\hspace{0.1cm}&\hspace{0.1cm}{0.558}&\hspace{0.1cm}0.688&\hspace{0.1cm}0.745&{0.387}  
        &\hspace{0.1cm}&\hspace{0.1cm}{0.451}&\hspace{0.1cm}0.572&\hspace{0.1cm}0.644&0.255\\
        
        CNN 
        &&\hspace{0.1cm}0.453&\hspace{0.1cm}{0.666}&\hspace{0.1cm}{0.787}&0.333   
        &\hspace{0.1cm}&\hspace{0.1cm}0.518&\hspace{0.1cm}{0.754}&\hspace{0.1cm}{0.841}&0.339  
        &\hspace{0.1cm}&\hspace{0.1cm}0.421&\hspace{0.1cm}{0.619}&\hspace{0.1cm}{0.755}&0.240\\
        
        Encoder 
        &&\hspace{0.1cm}0.382&\hspace{0.1cm}0.594&\hspace{0.1cm}0.684&0.269   &\hspace{0.1cm}&\hspace{0.1cm}0.533&\hspace{0.1cm}0.711&\hspace{0.1cm}0.777&0.344  &\hspace{0.1cm}&\hspace{0.1cm}0.411&\hspace{0.1cm}0.581&\hspace{0.1cm}0.674&0.243\\
        
        MT-LSTM 
        &&\hspace{0.1cm}{0.464}&\hspace{0.1cm}0.645&\hspace{0.1cm}0.713&0.357   &\hspace{0.1cm}&\hspace{0.1cm}0.555&\hspace{0.1cm}0.702&\hspace{0.1cm}0.743&0.346  &\hspace{0.1cm}&\hspace{0.1cm}0.440&\hspace{0.1cm}0.605&\hspace{0.1cm}0.706&{0.271}\\
        
        % ADLP &0.373&0.486&0.552&0.232   &0.479&0.581&0.660&0.266  &0.397&0.551&0.587&0.188\\
        
        S-CNN-LSTM 
        &&\hspace{0.1cm}0.228&\hspace{0.1cm}0.425&\hspace{0.1cm}0.575&0.170   &\hspace{0.1cm}&\hspace{0.1cm}0.446&\hspace{0.1cm}0.678&\hspace{0.1cm}0.759&0.282  &\hspace{0.1cm}&\hspace{0.1cm}0.328&\hspace{0.1cm}0.472&\hspace{0.1cm}0.572&0.115\\
        
        S-MT-CNN-LSTM 
        &&\hspace{0.1cm}0.356&\hspace{0.1cm}0.518&\hspace{0.1cm}0.593&0.192   &\hspace{0.1cm}&\hspace{0.1cm}0.453&\hspace{0.1cm}0.564&\hspace{0.1cm}0.682&0.251  &\hspace{0.1cm}&\hspace{0.1cm}0.368&\hspace{0.1cm}0.483&\hspace{0.1cm}0.618&0.178\\

        \hline

        BehaveMine (w/o Inter)
        &&\hspace{0.1cm}0.480&\hspace{0.1cm}0.663&\hspace{0.1cm}0.697&0.336   
        &\hspace{0.1cm}&\hspace{0.1cm}0.566&\hspace{0.1cm}0.736&\hspace{0.1cm}0.796&0.389  
        &\hspace{0.1cm}&\hspace{0.1cm}0.464&\hspace{0.1cm}0.617&\hspace{0.1cm}0.700&0.262\\

        BehaveMine (w/o Intra)
        &&\hspace{0.1cm}\underline{0.618}&\hspace{0.1cm}\underline{0.787}&\hspace{0.1cm}\underline{0.837}&\underline{0.453}   
        &\hspace{0.1cm}&\hspace{0.1cm}\underline{0.723}&\hspace{0.1cm}\underline{0.821}&\hspace{0.1cm}\underline{0.853}&\underline{0.513}  
        &\hspace{0.1cm}&\hspace{0.1cm}\underline{0.650}&\hspace{0.1cm}\textbf{0.808}&\hspace{0.1cm}\textbf{0.842}&\underline{0.394}\\
        
        \textbf{BehaveMine} 
        &&\hspace{0.1cm}\textbf{0.639}& \hspace{0.1cm}\textbf{0.813}&\hspace{0.1cm} \textbf{0.878}& \textbf{0.476}   
        &\hspace{0.1cm}&\hspace{0.1cm}\textbf{0.760}&\hspace{0.1cm}\textbf{0.870}&\hspace{0.1cm}\textbf{0.891}&\textbf{0.541}        &\hspace{0.1cm}&\hspace{0.1cm}\textbf{0.672}&\hspace{0.1cm}\underline{0.788}&\hspace{0.1cm}\underline{0.827}&\textbf{0.420}\\
        \hlineB{2}
    \end{tabular}
    \caption{Performance comparison on three real world datasets. For each row, the best performance and the second-best performance are highlighted in bold and underlined, respectively.}
    \label{table:comparison}
\end{table*}
