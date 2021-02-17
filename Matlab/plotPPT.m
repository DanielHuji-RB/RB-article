function plotPPT(fig,MontageType)


%% open TABLE
   
 % need to open PPT first
        toPPT('setPageFormat','16:9');
        templatePath = ['C:\']; % The template path has to be absolute!
        toPPT('applyTemplate',templatePath);
        
        %open png file and then send it to to ppt
    pwd='c:\tempfigs';
    cd(pwd)
    savefig('ppt_figure.fig')
    figPath = [pwd,'\ppt_figure.fig'];
    fig=openfig(figPath)
        toPPT(fig,'pos','E'); % NEH = NorthEastHalf%  'help getPosParameters'
        toPPT({['Montage Type: ',MontageType]},'SlideNumber','current');
 
end