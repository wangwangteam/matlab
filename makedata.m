% 获取路径下的所有.jpg文件，并生成filenames
        function [filenames,labels,data] = makedata(filedir)
            fileFolder=fullfile(filedir);
            dirOutput=dir(fullfile(fileFolder,'*.jpg'));
            filenames={dirOutput.name}';
            img=strcat(fileFolder,filenames);
            [s,~]=size(filenames);
            for i=1:s
                b=filenames{i};
                if strcmp(b(5:9),'daisy')
                    j=0;
                elseif  strcmp(b(5:9),'dande')
                    j=1;
                elseif strcmp(b(5:8),'rose')
                    j=2;
                elseif strcmp(b(5:9),'sunfl')
                    j=3;
                elseif strcmp(b(5:9),'tulip')
                    j=4;
                end
                m(i)=j;
            end
            labels=uint8(m');
            
            for i=1:s
                M=imread(img{i});
                k=imresize(M,[64,64]);
                r=k(:,:,1);
                g=k(:,:,2);
                b=k(:,:,3);
                J=reshape(r,[4096,1]);
                A=reshape(g,[4096,1]);
                L=reshape(b,[4096,1]);
                j=J';
                a=A';
                L=L';
                imgs=cat(2,j,a,L);
                data(i,:)=imgs;
            end
        end