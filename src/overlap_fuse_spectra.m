function [grid,Y,meta]=overlap_fuse_spectra(w,X,step)
%OVERLAP_FUSE_SPECTRA Merge overlapping spectrometer channels on one grid.
% X is samples x wavelength channels. Adjacent detector channels are
% blended with complementary raised-cosine weights over every overlap.
if nargin<3 || isempty(step), step=0.06; end
w=w(:); if size(X,2)~=numel(w), error('X columns must match wavelength vector.'); end
cuts=find(diff(w)<=0); starts=[1;cuts+1]; stops=[cuts;numel(w)]; ns=numel(starts);
grid=(min(w):step:max(w))'; ng=numel(grid); n=size(X,1);
num=zeros(n,ng); den=zeros(1,ng); rows=cell(max(ns-1,0),4);
for s=1:ns
    q=starts(s):stops(s); ws=w(q); weight=ones(ng,1); inside=grid>=ws(1)&grid<=ws(end);
    weight(~inside)=0;
    if s>1
        ov0=ws(1); ov1=w(stops(s-1));
        if ov1>ov0
            z=grid>=ov0 & grid<=ov1; t=(grid(z)-ov0)/(ov1-ov0);
            weight(z)=weight(z).*(0.5-0.5*cos(pi*t));
        end
    end
    if s<ns
        ov0=w(starts(s+1)); ov1=ws(end);
        if ov1>ov0
            z=grid>=ov0 & grid<=ov1; t=(grid(z)-ov0)/(ov1-ov0);
            weight(z)=weight(z).*(0.5+0.5*cos(pi*t));
            rows(s,:)={s,s+1,ov0,ov1};
        end
    end
    Yi=interp1(ws,X(:,q)',grid,'linear',NaN)'; Yi(:,weight==0)=0; Yi(~isfinite(Yi))=0;
    num=num+Yi.*weight'; den=den+weight';
end
den(den<eps)=1; Y=num./den;
meta=cell2table(rows,'VariableNames',{'Left_channel','Right_channel','Overlap_start_nm','Overlap_end_nm'});
end
