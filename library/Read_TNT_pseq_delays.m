function [DATA,PSEQ,filename] = Read_TNT_pseq_delays(filename, print_flag)
% This file opens, reads and parses *.tnt data files. The script is able to
% read in each section of the data file and stores them in appropriately
% named sections: TMAG,DATA,TMG2,PSEQ,SEQC,LOCV,MATV,TMG4,PEAK,TEQA,INTG,
% LNFT,CMNT,TMG3,TMG5,PGLB. The function will return the DATA and PSEQ
% data.
% 
% With print_flag = 1 (default), status messages are printed to the screen
% as a file is processed.
% 
% [DATA,[PSEQ]] = Read_TNT([filename], [print_flag]);

%% %%%%%%%%%%%%%%%%%%%%%% Revision History %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filename: Read_TNT.m
% Author:   Cris LaPierre <crislapi@nmr.mgh.harvard.edu>
% Date:     3 Aug 2012
% Descripton: This function replaces Read_Tecmag, Read_Raw_field,
%           Read_Tecmag_Header, and Read_Tecmag_hdr as well as most of 
%           data_recon_3D_undspled_3Drendering.m. It is the new default
%           program for reading in TNMR data files.
% 
% Revision: 2.1.1  (20 August 2014)
% Changes:  version 1.2.0
%           1) Modify capability of handiling undersample data sets.
%           Undersampling parameters and now written to TNMR as three
%           dashboard variables (undersampling ratio, 2nd Phase Encode and
%           3rd Phase Encode) and one table (signal_location). However,
%           these values can only be written if the variables already
%           exist. As a backup then, the variables are also saved to a
%           matfile signal_location_### where ### are the date and time of
%           creation. Finally, the mat file name is written to the comments
%           section of the data file. It is assumed this file will be in
%           the same folder as the data file.
%           2) Increased the options for determing data dimensions. Assumed
%           Pnts_2D is number of points in full data set and Acq_Pnts_2D is
%           number of collected points. If these don't match, data is
%           undersampled.
%           3) File first attempts to load variables, table. If it can't
%           load the table, it attempts to load the filename from the
%           comments. If it can't, for now it is assumed the data is not
%           undersampled and therefore signal_location was not created.
%           4) If there is an error, the script then tries to load all
%           parameters from the comments. If n_steps_GP or n_steps_GP2
%           still do not exists after this, the script reverts to the
%           original code for autodetermining them from gradient tables.
%           5) As a final check, if signal_location does not exist yet
%           acquired and actual data counts don't match OR signal_location
%           does exist but it doesn't match the data, prompt the user if
%           they would like to locate the file themselves. If they say yes,
%           run uigetfile so they can browse to the file (will open in same
%           directory as TNMR data file). If they cancel or if
%           signal_location still doesn't match data size or if they
%           clicked no, signal_location is automatically created.
% 
%           version 1.2.1 (24 Sep 2012)
%           1) Add prompt before autodetecting n_steps_GP and n_steps_GP2
%           to allow user to enter in values manually.
%           2) Ver 0 assumed signal location was in same folder as data
%           file. This version attempts to load from there but if it fails,
%           uigetfile is called before resorting to autodetection.
%           3) Fixes bug where signal_location file was never loaded if
%           all variables were pulled from comments section
% 
%           version 1.2.2 (21 Nov 2012)
%           1) New version of tNMR released and program could not read
%           Data files. Upgraded to ver 1.17 BIN in PSEQ section.
%           In the end, and extra byte (0) was being added to the end
%           of the first table (grise) in PSEQ section. For compatiability 
%           with our old and new data sets, an if statement was added in
%           the PSEQ section to read in one more byte and, if not zero, 
%           back up and begin reading in the next table.
% 
%           version 1.2.3 (09 Jan 2013)
%           1) Upgraded to tNMR3. Minor changes in PSEQ (now ver 1.18 BIN)  
%           that needed to be accounted for. Namely 24 bytes added after
%           sequence path but before num_fields. Also, in Acq field, units
%           have been added to data so can no longer use num2str. Switched
%           code to now read and write to structure as a cell.
%           2) Expanded the help to include an example function call
% 
%           version 1.2.4 (30 Jan 2013)
%           1) Added varargin to flag whether to print to screen or not
% 
%           version 1.2.5 (1 Apr 2013)
%           1) Set PSEQ.Comment to blank if not comment text. Previous
%           versions did not create the field if comment section was empty
%           but this lead to a bug when processing those files.
%           2) Modify logic in reshape_TNT to create dimensions of 1 if no
%           information on size is found. This helps enable processing of
%           FIDs.
% 
%           version 2.0 (16 May 2013)
%           1) We discovered a document containing the TNMR file format,
%           key to properly reading in the TNT data files. What we have is
%           a best guess. This version updates the file processing so that
%           it matches the key. PSEQ section has been modified since the
%           document was created. There are more bytes than it specifies.
%           For now we have not updated that section.
%           2) Added filename as output (7 May 2014)
% 
%           version 2.1 (28 July 2014)
%           1) Added code to reshape function to handle elastography data
%           sets. Main change: must add 'Acq=Dynamic' to comment of dynamic
%           acquisitions (multiple 3D images collected some delay appart).
%           With this data, a single file contains multiple images. Needed
%           to adjust how data is shapped. Used the channel dimension to
%           hold the image for each time delay.
%
%           version 2.1.1 (20 August 2014)
%           1) Modified file to correct for minor errors with UC Davis data
%-------------------------------------------------------------------------

DATA=[];
PSEQ=[];

if (nargin<1)
    [filename,pathname]=uigetfile({'*.tnt','TNMR data files(*.tnt)'},'Select file','MultiSelect','off');
    filename = fullfile(pathname,filename);
end
if (nargin<2)
    print_flag = 1;
end

if (exist(filename,'file')==2)
    s=dir(filename);
    if s.bytes>1056
        %% open file for reading
        fid=fopen(filename,'r','l');
        ver_nm = fread(fid,8,'*char')';
        

        %% Read in next section
        while ~feof(fid)

            section = upper(fread(fid,4,'*char')');
            switch section
                case 'TMAG'
                    %% Read TMAG portion (Header)
                    % Code borrowed from Read_Tecmag_Header. File Read_Tecmag_hdr 
                    % contains the information on variable names, format and 
                    % size needed to properly interpret the file header,
                    % which consists of the first 1044 bytes in the file
                    flag = logical(fread(fid,1,'int'));
                    if flag
                        % size in bytes of this section. Generally 1024.
                        bytes = fread(fid,1,'int');
                        offset = ftell(fid) + bytes;
                        
                        % Create fields in PSEQ.Dashboard now so they are 
                        % in the desired order
                        PSEQ.Dashboard.Acquisition.Nucleus = [];
                        PSEQ.Dashboard.Acquisition.Acq_Pnts = [];
                        PSEQ.Dashboard.Acquisition.Pnts_1D = [];
                        PSEQ.Dashboard.Acquisition.SW = [];
                        PSEQ.Dashboard.Acquisition.Filter = [];
                        PSEQ.Dashboard.Acquisition.Dwell_Time = [];
                        PSEQ.Dashboard.Acquisition.Acq_Time = [];
                        PSEQ.Dashboard.Acquisition.Last_Delay = [];
                        PSEQ.Dashboard.Acquisition.Scans_1D = [];
                        PSEQ.Dashboard.Acquisition.Act_Scans_1D = [];
                        PSEQ.Dashboard.Acquisition.Scan_Start_1D = [];
                        PSEQ.Dashboard.Acquisition.Repeat_Times = [];
                        PSEQ.Dashboard.Acquisition.SA_Dim = [];
                        PSEQ.Dashboard.Acquisition.Dummy_Scans = [];
                        PSEQ.Dashboard.Acquisition.Grd_Orient = [];
                        PSEQ.Dashboard.Acquisition.Pnts_2D = [];
                        PSEQ.Dashboard.Acquisition.Act_Pnts_2D = [];
                        PSEQ.Dashboard.Acquisition.Pnts_Start_2D = [];
                        PSEQ.Dashboard.Acquisition.Pnts_3D = [];
                        PSEQ.Dashboard.Acquisition.Act_Pnts_3D = [];
                        PSEQ.Dashboard.Acquisition.Pnts_Start_3D = [];
                        PSEQ.Dashboard.Acquisition.Pnts_4D = [];
                        PSEQ.Dashboard.Acquisition.Act_Pnts_4D = [];
                        PSEQ.Dashboard.Acquisition.Pnts_Start_4D = [];
                        PSEQ.Dashboard.Acquisition.SW_2D = [];
                        PSEQ.Dashboard.Acquisition.SW_3D = [];
                        PSEQ.Dashboard.Acquisition.SW_4D = [];
                        PSEQ.Dashboard.Acquisition.Dwell_2D = [];
                        PSEQ.Dashboard.Acquisition.Dwell_3D = [];
                        PSEQ.Dashboard.Acquisition.Dwell_4D = [];
                        PSEQ.Dashboard.Acquisition.Grd_Theta = [];
                        PSEQ.Dashboard.Acquisition.Grd_Phi = [];
                                                
                        % Read in header
                        Hdr_var=Read_TNT_hdr;

                        Hdr_name=Hdr_var(:,1:28);
                        Hdr_offset=Hdr_var(:,29:34);
                        Hdr_type=Hdr_var(:,36:43);
                        Hdr_size=Hdr_var(:,45:47);
                        Hdr_desc=Hdr_var(:,48:85);
                        % function strrep pour enlever les espaces.

                        Number_var=size(Hdr_name);

                        for i=1:Number_var(1,1)
                            Var_name=strrep(Hdr_name(i,:),' ','');
                            Var_offset=eval(Hdr_offset(i,:));
                            Var_type=strrep(Hdr_type(i,:),' ','');
                            Var_size=eval(Hdr_size(i,:));
                            Var_desc=Hdr_desc(i,:);

                            fseek(fid,Var_offset,'bof');
                            Var_data=fread(fid,Var_size,Var_type)';
                            if (strcmp(Var_type,'char'))&&((Var_size-1)>0)
                                Var_data=char(Var_data);
                            end
                            eval(strcat('PSEQ.Dashboard.',Var_name,'=Var_data;'));
                        end
                    end
                    % position file identifier at start of next section
                    fseek(fid,offset,'bof');

                case 'DATA'
                    %% Read in waveform data
                    % Waveform data immediately follows the header. They
                    % are complex floats, meaning a single number is made
                    % up of 8 bytes of data - 4 for the real part and 4 for
                    % the imaginary part.
                    % Bytes 1048-1052 appear to be a flag indicating if there
                    % is data included. If yes, the next 4 bytes indicate the
                    % total size in bytes of the unformatted waveform data.
                    flag = logical(fread(fid,1,'int'));
                    if flag
                        bytes = fread(fid,1,'int');

                        % div 4 because 4 bytes per float
                        % div 2 because 2 rows x indicated size
                        DATA=fread(fid,[2,bytes/4/2],'float32'); 
                        DATA=complex(DATA(1,:),DATA(2,:));
                    end
                    
                case 'TMG2'
                    %% The first 4 bytes appear to be a flag indicating if there
                    % are entries. If yes, the next 4 bytes indicate the
                    % total size in bytes of this section.
                    % Unsure of the format so currently everything is read
                    % in as integers.
                    flag = logical(fread(fid,1,'int'));
                    if flag
                        bytes = fread(fid,1,'int');
                        offset = ftell(fid);
                        % TMG2 = fread(fid,bytes/4,'int')';

                        % Read in settings
                        TMG2_var=Read_TMG2;

                        TMG2_name=TMG2_var(:,1:28);
                        TMG2_offset=TMG2_var(:,29:34);
                        TMG2_type=TMG2_var(:,36:43);
                        TMG2_size=TMG2_var(:,45:47);
                        TMG2_desc=TMG2_var(:,48:85);
                        % function strrep pour enlever les espaces.

                        Number_var=size(TMG2_name);

                        for i=1:Number_var(1,1)
                            Var_name=strrep(TMG2_name(i,:),' ','');
                            Var_offset=eval(TMG2_offset(i,:)) + offset;
                            Var_type=strrep(TMG2_type(i,:),' ','');
                            Var_size=eval(TMG2_size(i,:));
                            Var_desc=TMG2_desc(i,:);

                            fseek(fid,Var_offset,'bof');
                            Var_data=fread(fid,Var_size,Var_type)';
                            if (strcmp(Var_type,'char'))&&((Var_size-1)>0)
                                Var_data=char(Var_data);
                            end
                            eval(strcat('TMG2.',Var_name,'=Var_data;'));
                        end
                    end
                
                case 'PSEQ'
                    %% This field contains the image seen graphically in
                    % TNMR on the Sequence screen. The data is displayed
                    % with each field on its own row and columns indicating
                    % events during the sequence. 2D and 3D tables do not
                    % appear to be written to the file though any
                    % associated tables are. The variable names may be used
                    % to locate the desired table(s).
                    entry_bytes = 60;

                    flag = logical(fread(fid,1,'int'));
                    if flag
                        rev = fread(fid,8,'*char')';                      
                        
                        % May be a boolean, but as long as it is 0, this is
                        % statement accomplishes the same thing.
                        filename_len = fread(fid,1,'int');
                        file_name = fread(fid,filename_len,'*char')';
                        
                        % New addition on 1/9/13 for TNMR3. Writing 24 new
                        % bytes of data to the PSEQ heading.
                        if strcmpi('1.18 BIN',rev)
                            sz = fread(fid,1,'int');
                            sz = fread(fid,1,'int');
                            sz = fread(fid,1,'int');
                            uknw_field = fread(fid,sz,'*char');
                            sz = fread(fid,1,'int');
                            uknw_field = fread(fid,sz,'*char');
                        end
                            
                        num_fields = fread(fid,1,'int'); % Rows
                        num_events = fread(fid,1,'int'); % event1 is column of field names

                        % Sequence Rows
                        for field = 1:num_fields
                            num_local_events = fread(fid,1,'int');
                            % unknown settings, strings
                            address = fread(fid,1,'int')';
                            BitLength = fread(fid,1,'int');
                            % The next setting corresponds to the variable
                            % type: HP=phase table, HS=GradWaveformTbl, 
                            % 2G=GradAmpTbl (pre acq), 3G=GradAmpTbl (post acq)
                            % 14=DelayTbl, 10 = ?
                            icon_library_type = fread(fid,4,'*char')';
                            visible_flag = fread(fid,1,'int');
                            private_data = fread(fid,1,'int');
                            group = fread(fid,1,'int');
                            
                            % Sequence Events (columns)
                            for event = 1:num_local_events
                                switch event
                                    case 1
                                        % Each event starts with a string (usually 0)
                                        default_str_len = fread(fid,1,'int');
                                        default_str = fread(fid,default_str_len,'*char')';
                                        % First event also contains field name (2x)
                                        label_str_len = fread(fid,1,'int');
                                        label_str = fread(fid,label_str_len,'*char')';

                                        % First event contains the row name
                                        data_str_len = fread(fid,1,'int');
                                        field_nm = format_name(fread(fid,data_str_len,'*char')'); % remove periods/colons, replace space with "_"

                                        % skip remaining sub-entries, at least until they can be
                                        % decoded
                                        fseek(fid,entry_bytes - 4,'cof');                                      
                                    otherwise
                                        % There are 15 clusters of 4 bytes for each event. 
                                        % Not all have been decoded.
                                        for bytes = 1:entry_bytes/4
                                            nm_size = fread(fid,1,'int');
                                            if nm_size > 0
                                                switch bytes
                                                    case 1
                                                        %% Value/name that appears on TNMR pulse sequence screen
                                                        eval(['PSEQ.Sequence.' field_nm '.Setting(event-1) = {fread(fid,nm_size,''*char'')''};']);
                                                    case 2
                                                        %% Appears to contain variable names for gradient waveform tables
                                                        eval(['PSEQ.Sequence.' field_nm '.GradWaveTbl(event-1) = {fread(fid,nm_size,''*char'')''};']);                                                    
                                                    case 4
                                                        %% Appears to contain variable names for phase tables
                                                        eval(['PSEQ.Sequence.' field_nm '.PhaseTbl(event-1) = {fread(fid,nm_size,''*char'')''};']);                                                    
                                                    case {6,8}
                                                        %% Appears to contain variable names for Gradient Amplitude, Phase Tables
                                                        eval(['PSEQ.Sequence.' field_nm '.Delay_Branch_GradAmpTbl(event-1) = {fread(fid,nm_size,''*char'')''};']);
                                                    case 10
                                                        %% Appears to contain variable names for delay tables
                                                        eval(['PSEQ.Sequence.' field_nm '.DelayTbl(event-1) = {fread(fid,nm_size,''*char'')''};']);                                                    
                                                    case 15
                                                        % Appears to contain acquisition properties for associated channel
                                                        switch nm_size
                                                            case {1,'1'}
                                                                sz=fread(fid,1,'int');
                                                                eval(['PSEQ.Sequence.' field_nm '.acq_pnts_Hz(event-1) = str2num(fread(fid,sz,''*char'')'');']);
                                                                sz=fread(fid,1,'int');
%                                                                 eval(['PSEQ.Sequence.' field_nm '.sweep(event-1) = str2num(fread(fid,sz,''*char'')'');']);
                                                                eval(['PSEQ.Sequence.' field_nm '.sweep(event-1) = {fread(fid,sz,''*char'')''};']);
                                                                sz=fread(fid,1,'int');
%                                                                 eval(['PSEQ.Sequence.' field_nm '.filter(event-1) = str2num(fread(fid,sz,''*char'')'');']);
                                                                eval(['PSEQ.Sequence.' field_nm '.filter(event-1) = {fread(fid,sz,''*char'')''};']);
                                                                sz=fread(fid,1,'int');
%                                                                 eval(['PSEQ.Sequence.' field_nm '.dwell_sec(event-1) = str2num(fread(fid,sz,''*char'')'');']);
                                                                eval(['PSEQ.Sequence.' field_nm '.dwell_sec(event-1) = {fread(fid,sz,''*char'')''};']);
                                                                sz=fread(fid,1,'int');
%                                                                 eval(['PSEQ.Sequence.' field_nm '.acq_tm_sec(event-1) = str2num(fread(fid,sz,''*char'')'');']);
                                                                eval(['PSEQ.Sequence.' field_nm '.acq_tm_sec(event-1) = {fread(fid,sz,''*char'')''};']);
                                                                eval(['PSEQ.Sequence.' field_nm '.dash_link(event-1) = fread(fid,1,''int'');']);
                                                                fseek(fid,2,'cof'); % unknown bytes at end of acquisition properties stream
                                                        end
                                                    otherwise
                                                        %% does NOT indicate the number of chars to read in. It is the desired value
                                                        % Since correspoinding setting is not known, for now do nothing.

                                                end % switch bytes
                                            end % if nm_size...
                                        end % for bytes...
                                end % switch event...
                            end % for event...
                        end % for field...
                    end % if PSEQ_flag
                    
                    %% Read in unknown table
                    sz = fread(fid,1,'int');
                    fread(fid,sz,'int');

                    %% Read in variables
                    entry_bytes = 76;
                    num_vars = fread(fid,1,'int');
                    for var = 1:num_vars
                        nm_size = fread(fid,1,'int');
%                         PSEQ.Table(var).name = {fread(fid,nm_size,'*char')'};
                        table_nm = format_name(fread(fid,nm_size,'*char')');

                        d_size = fread(fid,1,'int');
                        stream = fread(fid,d_size,'*char')';
                        % parse string7
                        temp = textscan(stream,'%s','delimiter',' \r\n','MultipleDelimsAsOne', 1);
%                         PSEQ.Table(var).val = [temp{:}];
                        eval(['PSEQ.Table.' table_nm '.val = [temp{:}];']);

                        % Acquisition properties
                        sz = fread(fid,1,'int'); % + Add
                        eval(['PSEQ.Table.' table_nm '.IncSch_add = {fread(fid,sz,''*char'')''};']);
                        sz = fread(fid,1,'int'); % 3m
                        eval(['PSEQ.Table.' table_nm '.IncVal = {fread(fid,sz,''*char'')''};']);
                        sz = fread(fid,1,'int'); % Every pass
                        eval(['PSEQ.Table.' table_nm '.IncSch_pass = {fread(fid,sz,''*char'')''};']);
                        fread(fid,1,'int');
                        % Indicator of the type of table
                        str = fread(fid,4,'*char')';
                        dim = fread(fid,1,'int');
                        switch str(1:2)
                            case 'HS'
                                str = 'Gradient Waveform Table';
                            case '2G'
                                str = [num2str(dim) 'D Gradient Amplitude Table (pre-acquisition)'];
                            case '3G'
                                str = [num2str(dim) 'D Gradient Amplitude Table (post-acquisition)'];
                            case 'HP'
                                str = [num2str(dim) 'D Phase Table'];
                            otherwise
                                fseek(fid,-8,'cof');
                                num = fread(fid,1,'int');
                                fseek(fid,4,'cof');
                                if num == 1
                                    str = [num2str(dim) 'D Delay Table'];
                                elseif num == 10
                                    display(str);
                                    pause
                                elseif num == 14
                                    str = [num2str(dim) 'D Branch Table'];
                                end
                                
                        end
                        eval(['PSEQ.Table.' table_nm '.type = {str};']);
                                
                        eval(['PSEQ.Table.' table_nm '.steps = fread(fid,1,''int'');']);
                        fread(fid,1,'int');
                        fread(fid,1,'int');
                        fread(fid,1,'int');
                        fread(fid,1,'int');
                        fread(fid,1,'int');
                        sz = fread(fid,1,'int');
                        fread(fid,sz,'*char')';
                        sz = fread(fid,1,'int');
                        fread(fid,sz,'*char')';
                        sz = fread(fid,1,'int');
                        fread(fid,sz,'*char')';
                        fread(fid,1,'int');
                        fread(fid,1,'int');
                        
                        % Nov 21, 2012: New version of tNMR adds an extra 
                        % byte at the end of the grise table. This
                        % interferes with the read-in order and shifts the
                        % reading frame, crashing the read. This code reads
                        % in one byte past where it used to and checks if
                        % it is zero. If it is not, it is assumed to be the
                        % start of the next variable. The if statement
                        % backs up 4 bytes and begins reading the new table
                        unkw = fread(fid,1,'int');
                        if unkw > 0
                            fseek(fid,-4,'cof');
                        end
                    end
                    
                    %% Subsection: Sequence
                    flag = logical(fread(fid,1,'int'));
                    if flag
                        %% Read in section name
                        nm_size = fread(fid,1,'int');
                        fread(fid,nm_size,'*char'); % Sequence

                        num_entries = fread(fid,1,'int');

                        % Read in setting names
                        for entry = 1:num_entries
                            nm_size = fread(fid,1,'int');
                            var_nm = format_name(fread(fid,nm_size,'*char')');
%                             PSEQ.Dashboard.name(entry) = {fread(fid,nm_size,'*char')'};
                            eval(['PSEQ.Dashboard.Sequence.' var_nm ' = [];']);
                        end

                        %% Read in dashboard vals
                        % bytes 1-4,37-40:number of characters in name
                        % bytes 5-8:   unknown
                        % bytes 9-12,41-44:number of characters in value
                        % bytes 13-16: integer - 9?
                        % bytes 17-20: lower range? (100m/0.5u/-100000000.0Hz)
                        % bytes 21-24: upper range? (1024s/100000000.0Hz)
                        % bytes 25-28: unknown (0/1)
                        % bytes 29-32: unknown (0)
                        % bytes 33-36: unknown (0)
                        %####### see 1,3 for jump
                        % bytes 45-48: unknown (0)
                        % bytes 49-52: unknown (0)
                        % bytes 53-56: unknown (0)
                        % bytes 57-60: field/row in pulse sequence where used
                        entry_bytes = 60;

                        num_entries = fread(fid,1,'int');

                        for entry = 1:num_entries
                            % bytes 1-4
                            nm_size = fread(fid,1,'int');
                            var_nm = format_name(fread(fid,nm_size,'*char')');

                            % bytes 5-8
                            sz = fread(fid,1,'int');
                            fread(fid,sz,'*char');

                            % bytes 9-12
                            sz = fread(fid,1,'int');
%                             PSEQ.Dashboard.value(strcmp(PSEQ.Dashboard.name,str)) = {fread(fid,sz,'*char')'};
                            eval(['PSEQ.Dashboard.Sequence.' var_nm ' = {fread(fid,sz,''*char'')''};']);

                            % bytes 13-16
                            fread(fid,1,'int');
                            % bytes 17-20
                            sz = fread(fid,1,'int');
                            fread(fid,sz,'*char');
                            % bytes 21-24
                            sz = fread(fid,1,'int');
                            fread(fid,sz,'*char');
                            % bytes 25-28
                            fread(fid,1,'int');
                            % bytes 29-32
                            fread(fid,1,'int');
                            % bytes 33-36
                            fread(fid,1,'int');

                            % bytes 37-40
                            sz = fread(fid,1,'int');
                            fread(fid,sz,'*char');
                            % bytes 41-44
                            sz = fread(fid,1,'int');
                            fread(fid,sz,'*char');

                            % bytes 45-48
                            fread(fid,1,'int');
                            % bytes 49-52
                            fread(fid,1,'int');
                            % bytes 53-56
                            fread(fid,1,'int');
                            % bytes 57-60
%                             PSEQ.Dashboard.field(strcmp(PSEQ.Dashboard.name,str)) = fread(fid,1,'int'); % confusing. Don't add.
                            fread(fid,1,'int');
                        end
                    end 

                case 'SEQC'
                    %% Appears to contain a single entry indicatting the sequence
                    nm_size = fread(fid,1,'int');
                    PSEQ.SEQC = {fread(fid,nm_size,'*char')'};

                case 'LOCV'
                    %% First 4 bytes after section tag indicate the number of
                    % entries in this section. The length of each entry is
                    % 60 bytes. The end of each one is flagged with [255 255 255 255].
                    % These appear to be user-defined variables on the Sequence 
                    % tab in the dashboard (blue)
                    % bytes 1-4:   number of characters in name
                    % bytes 5-8:   unknown
                    % bytes 9-12:  number of characters in value
                    % bytes 13-16: unknown (0)
                    % bytes 17-20: unknown (0)
                    % bytes 21-24: unknown (0)
                    % bytes 25-28: unknown (0)
                    % bytes 29-32: unknown (0)
                    % bytes 33-36: unknown (0)
                    % bytes 37-40: number of characters in name
                    % bytes 41-44: unknown (0)
                    % bytes 45-48: unknown (0)
                    % bytes 49-52: unknown (0)
                    % bytes 53-56: field/row in pulse sequence where used
                    % bytes 57-60: end flag [255 255 255 255]
                    end_flag = [char(255) char(255) char(255) char(255)];
                    num_entries = fread(fid,1,'int');

                    for entry = 1:num_entries
                        % Variable name
                        nm_size = fread(fid,1,'int');
%                         PSEQ.Dashboard.name(end+1) = {fread(fid,nm_size,'*char')'};
                        var_nm = format_name(fread(fid,nm_size,'*char')');

                        % Unknown entry
                        sz = fread(fid,1,'int');
                        fread(fid,sz,'*char');

                        % Associated value
                        d_size = fread(fid,1,'int');
%                         PSEQ.Dashboard.value(end+1) = {fread(fid,d_size,'*char')'};
                        eval(['PSEQ.Dashboard.Sequence.' var_nm ' = {fread(fid,d_size,''*char'')''};']);

                        while true
                            sz = fread(fid,1,'int');
                            uknw_val = fread(fid,sz,'*char')';
                            if sz>0 && sz<5 && strcmp(uknw_val,end_flag(1:sz))
                                fread(fid,4-sz,'*char');
                                break
                            end
                        end
                    end

                case 'MATV'
                    %% First 4 bytes after section tag indicate the number of
                    % entries in this section. This section appears to just contain
                    % a list of variable names and their value.
                    % All these values are already entered in PSEQ.Dashboard.
                    num_entries = fread(fid,1,'int');

                    for entry = 1:num_entries
                        nm_size = fread(fid,1,'int');
                        MATV.name(entry) = {fread(fid,nm_size,'*char')'};

                        d_size = fread(fid,1,'int');
                        MATV.val(entry) = {fread(fid,d_size,'*char')'};
                    end

                case 'TMG4'
                    %% This section begins with a stream flag (1=stream, 0=
                    % no stream) followed by an integer indicating the 
                    % number of bytes included in this section.
                    
                    flag = logical(fread(fid,1,'int'));
                    if flag
                        bytes = fread(fid,1,'int');
                        for entry = 1:bytes/4
                            TMG4(entry) = fread(fid,1,'int');
                        end
                    end

                case 'PEAK'
                    %% This section begins with a stream flag (1=stream, 0=
                    % no stream) followed by an integer indicating the 
                    % number of bytes included in this section.
                    
                    flag = logical(fread(fid,1,'int'));
                    if flag
                        bytes = fread(fid,1,'int');
                        for entry = 1:bytes/4
                            PEAK(entry) = fread(fid,1,'int');
                        end
                    end
                    
                case 'TEQA'
                    %% This section begins with a stream flag (1=stream, 0=
                    % no stream) followed by an integer indicating the 
                    % number of bytes included in this section.
                    
                    flag = logical(fread(fid,1,'int'));
                    if flag
                        bytes = fread(fid,1,'int');
                        for entry = 1:bytes/4
                            TEQA(entry) = fread(fid,1,'int');
                        end
                    end
                    
                case 'INTG'
                    %% This section begins with a stream flag (1=stream, 0=
                    % no stream) followed by an integer indicating the 
                    % number of entries included in this section. Each
                    % entry appears to be allocated 48 bytes
                    entry_bytes = 48;
                    
                    flag = logical(fread(fid,1,'int'));
                    if flag
                        entries = fread(fid,1,'int');
                        for entry = 1:entries
                            for byte = 1:entry_bytes/4
                                INTG(entry).val(byte) = fread(fid,1,'int');
                            end
                        end
                    end
                    
                case 'LNFT'
                    %% This section begins with a stream flag (1=stream, 0=
                    % no stream) followed by an integer indicating the 
                    % number of bytes included in this section.
                    
                    flag = logical(fread(fid,1,'int'));
                    if flag
                        bytes = fread(fid,1,'int');
                        for entry = 1:bytes/4
                            LNFT(entry) = fread(fid,1,'int');
                        end
                    end
                    
                case 'CMNT'
                    %% Sequence comments
                    % This section begins with a stream flag (1=stream, 0=
                    % no stream) followed by an integer indicating the 
                    % number of bytes included in this section.
                    
                    flag = logical(fread(fid,1,'int'));
                    if flag
                        bytes = fread(fid,1,'int');
                        stream = fread(fid,bytes,'*char')';
                        % parse string
                        Comment = textscan(stream,'%s','delimiter','\r\n','MultipleDelimsAsOne', 1);
                        PSEQ.Comment = Comment{1};
                    else
                        PSEQ.Comment = {''};
                    end

                case 'TMG3'
                    %% Unsure what values and formats the data is stored here. It
                    % appears to be all numeric but unsure of the encoding (float?)
                    entry_bytes = 520;

                    fread(fid,1,'int');
                    num_entries = fread(fid,1,'int');

                    for entry = 1:num_entries
                        for byte = 1:entry_bytes/4
                            TMG3(entry).val(byte) = fread(fid,1,'int');
                        end
                    end

                case 'TMG5'
                    %% unsure what this section contains
                    entry_bytes = 152;

                    flag = logical(fread(fid,1,'int'));
                    if flag
                        num_entries = fread(fid,1,'int');

                        for entry = 1:num_entries
                            for byte = 1:entry_bytes/4
                                TMG5(entry).val(byte) = fread(fid,1,'int');
                            end
                        end
                    end
                    
                case 'PGLB'
                    %% appears to contain a single boolean. Is false in our files.
                    % Perhaps if true, additional data is included similar to what
                    % occurs in TMG4 to include the Sequence Comments
                    fread(fid,1,'int');

            end % switch section
        end    % while ~feof(...
        %% close file
        fclose(fid);
        
    end % if s.bytes...
end % if exist(filename...

if exist('DATA','var') && exist('PSEQ','var')
    % Identify file to screen for printout of channels, steps.
    [pathstr, name, ext] = fileparts(filename);
    if print_flag, fprintf('\n*** File: %s\n',name); end
end

end % Function Read_TNT


function good_var_name = format_name(potential_var_name)
% The purpose of this function is to format string names so they can be
% used as variable names and structure fields. It removes punctuation and
% replaces spaces with underscores.

good_var_name = regexprep(strtrim(potential_var_name),{'[()\.:+-]','[ /]'},{'','_'});
if ~isvarname(good_var_name)
    % Likely starts with a number
    good_var_name = ['a_' good_var_name];
end

end

function [M]=Read_TNT_hdr()
M=[% Number of points and scans in all dimensions
   'Acquisition.Pnts_1D        ',' 000020 ',' long   ',' 001 ',' Points requested - 1D               ';
   'Acquisition.Pnts_2D        ',' 000024 ',' long   ',' 001 ',' Points requested - 2D               ';
   'Acquisition.Pnts_3D        ',' 000028 ',' long   ',' 001 ',' Points requested - 3D               ';
   'Acquisition.Pnts_4D        ',' 000032 ',' long   ',' 001 ',' Points requested - 4D               ';
   
   'Acquisition.Act_Pnts_1D    ',' 000036 ',' long   ',' 001 ',' Points completed - 1D               ';
   'Acquisition.Act_Pnts_2D    ',' 000040 ',' long   ',' 001 ',' Points completed - 2D               ';
   'Acquisition.Act_Pnts_3D    ',' 000044 ',' long   ',' 001 ',' Points completed - 3D               ';
   'Acquisition.Act_Pnts_4D    ',' 000048 ',' long   ',' 001 ',' Points completed - 4D               ';
    
   'Acquisition.Acq_Pnts       ',' 000052 ',' long   ',' 001 ',' Number of points to acquire         ';
   
   'Acquisition.Scan_Start_1D  ',' 000056 ',' long   ',' 001 ',' scan/pt to start acquisition - 1D   ';
   'Acquisition.Pnts_Start_2D  ',' 000060 ',' long   ',' 001 ',' scan/pt to start acquisition - 2D   ';
   'Acquisition.Pnts_Start_3D  ',' 000064 ',' long   ',' 001 ',' scan/pt to start acquisition - 3D   ';
   'Acquisition.Pnts_Start_4D  ',' 000068 ',' long   ',' 001 ',' scan/pt to start acquisition - 4D   ';
   
   'Acquisition.Scans_1D       ',' 000072 ',' long   ',' 001 ',' Scans 1D requested                  ';
   'Acquisition.Act_Scans_1D   ',' 000076 ',' long   ',' 001 ',' Scans 1D completed                  ';
   'Acquisition.Dummy_Scans    ',' 000080 ',' long   ',' 001 ',' Num scans prior to collecting       ';
   
   'Acquisition.Repeat_Times   ',' 000084 ',' long   ',' 001 ',' Number of times to repeat scan      ';
   'Acquisition.SA_Dim         ',' 000088 ',' long   ',' 001 ',' Signal averaging dimension          ';
   'Acquisition.SA_mode        ',' 000088 ',' long   ',' 001 ',' Behavior of signal averager         ';
   
%    'space1                     ',' 000096 ',' char   ',' 000 ','                                     ';

% Field and frequencies
   'magnet_field               ',' 000096 ',' float64',' 001 ',' Magnetic field                      ';
   'Frequency.F1_Freq          ',' 000104 ',' float64',' 001 ',' Obverve frequency Rec1              ';
   'Frequency.F2_Freq          ',' 000112 ',' float64',' 001 ',' Obverve frequency Rec2              ';
%    'Frequency.F3_Freq          ',' 000120 ',' float64',' 001 ',' Obverve frequency Rec3              ';
%    'Frequency.F4_Freq          ',' 000128 ',' float64',' 001 ',' Obverve frequency Rec4              ';
%    'Frequency.R1_base_freq     ',' 000136 ',' float64',' 001 ',' Base frequency Rec1                 ';
%    'Frequency.R2_base_freq     ',' 000144 ',' float64',' 001 ',' Base frequency Rec2                 ';
%    'Frequency.R3_base_freq     ',' 000152 ',' float64',' 001 ',' Base frequency Rec3                 ';
%    'Frequency.R4_base_freq     ',' 000160 ',' float64',' 001 ',' Base frequency Rec4                 ';
%    'Frequency.R1_offset_freq   ',' 000168 ',' float64',' 001 ',' R1 offset from base                 ';
%    'Frequency.R2_offset_freq   ',' 000176 ',' float64',' 001 ',' R2 offset from base                 ';
%    'Frequency.R3_offset_freq   ',' 000184 ',' float64',' 001 ',' R3 offset from base                 ';
%    'Frequency.R4_offset_freq   ',' 000192 ',' float64',' 001 ',' R4 offset from base                 ';
%    'ref_freq                   ',' 000200 ',' float64',' 001 ',' Ref freq for axis calculation       ';

%    'NMR_freq                   ',' 000208 ',' float64',' 001 ',' Absolute NMR frequency              ';
%    'obs_channel                ',' 000216 ',' short  ',' 001 ',' observe channel                     ';
%    'spaces2                    ',' 000218 ',' char   ',' 042 ','                                     ';

% spectral width, dwell and filter
   'Acquisition.SW             ',' 000260 ',' float64',' 001 ',' spectral width in Hz                ';
   'Acquisition.SW_2D          ',' 000268 ',' float64',' 001 ',' spectral width in Hz - 2D           ';
   'Acquisition.SW_3D          ',' 000276 ',' float64',' 001 ',' spectral width in Hz - 3D           ';
   'Acquisition.SW_4D          ',' 000284 ',' float64',' 001 ',' spectral width in Hz - 4D           ';
   'Acquisition.Dwell_Time     ',' 000292 ',' float64',' 001 ',' dwell time (s)                      ';
   'Acquisition.Dwell_2D       ',' 000300 ',' float64',' 001 ',' dwell time (s) - 2D                 ';
   'Acquisition.Dwell_3D       ',' 000308 ',' float64',' 001 ',' dwell time (s) - 3D                 ';
   'Acquisition.Dwell_4D       ',' 000316 ',' float64',' 001 ',' dwell time (s) - 4D                 ';
   'Acquisition.Filter         ',' 000324 ',' float64',' 001 ',' filter                              ';
   'Misc.Exp_Elapsed_Time      ',' 000332 ',' float64',' 001 ',' time for whole experiment           ';
   'Acquisition.Acq_Time       ',' 000340 ',' float64',' 001 ',' acquisition time                    ';

   'Acquisition.Last_Delay     ',' 000348 ',' float64',' 001 ',' last delay in seconds               ';

%    'spectrum_direction         ',' 000356 ',' short  ',' 001 ',' 1 or -1                             ';
%    'hardware_sideband          ',' 000358 ',' short  ',' 001 ','                                     ';
%    'Taps                       ',' 000360 ',' short  ',' 001 ',' num of taps on receiver filter      ';
%    'Type                       ',' 000362 ',' short  ',' 001 ',' type of filter                      ';
%    'dDigRec                    ',' 000364 ',' ubit4  ',' 001 ',' toggle for digital receiver         ';
%    'nDigitalCenter             ',' 000368 ',' long   ',' 001 ',' num shift pnts for digital receiver ';
%    'spaces3                    ',' 000372 ',' char   ',' 016 ','                                     ';

% Hardware settings
%    'tramsmitter_gain           ',' 000388 ',' short  ',' 001 ',' transmitter gain                    ';
%    'receiver_gain              ',' 000390 ',' short  ',' 001 ',' receiver gain                       ';
   'MultiRec.NumRx             ',' 000392 ',' short  ',' 001 ',' Number of Rx in MultiRx system      ';
%    'RG2                        ',' 000394 ',' short  ',' 001 ',' receiver gain Rx channel 2          ';
%    'receiver_phase             ',' 000396 ',' float64',' 001 ',' receiver phase                      ';
%    'spaces4                    ',' 000404 ',' char   ',' 004 ','                                     ';
%
% spinning speed information
%    'set_spin_rate              ',' 000408 ',' ushort ',' 001 ',' set spin rate                       ';
%    'actual_spin_rate           ',' 000410 ',' ushort ',' 001 ',' actual spin rate read from meter    ';
%
% Lock information
%    'lock_field                 ',' 000412 ',' short  ',' 001 ',' lock field value                    ';
%    'lock_power                 ',' 000414 ',' short  ',' 001 ',' lock transmitter power              ';
%    'lock_gain                  ',' 000416 ',' short  ',' 001 ',' lock receiver gain                  ';
%    'lock_phase                 ',' 000418 ',' short  ',' 001 ',' lock phase                          ';
%    'lock_freq_mhz              ',' 000420 ',' float64',' 001 ',' lock frequency in MHz               ';
%    'lock_ppm                   ',' 000428 ',' float64',' 001 ',' lock ppm                            ';
%    'H2O_freq_ref               ',' 000436 ',' float64',' 001 ',' H1 freq of H2O                      ';
%    'spaces5                    ',' 000444 ',' char   ',' 016 ','                                     ';
%
% VT information
%    'set_temperature            ',' 000460 ',' float64',' 001 ',' set temperature                     ';
%    'actual_temperature         ',' 000468 ',' float64',' 001 ',' actual temperature                  ';
%
% Shim information
%    'shim_unit                  ',' 000476 ',' float64',' 001 ',' shim units                          ';
%    'shims                      ',' 000484 ',' short  ',' 036 ',' shim values                         ';
%    'shims_FWHM                 ',' 000556 ',' float64',' 001 ',' full width at half max             ';
%
% Hardware specific information
%    'HH_dcpl_attn               ',' 000564 ',' short  ',' 001 ',' decoupler attenuation               ';
%
%    'DF_DN                      ',' 000566 ',' short  ',' 001 ',' decoupler                           ';
%    'F1_tran_mode               ',' 000568 ',' short  ',' 007 ',' F1 Pulse transmitter switches       ';
%    'dec_BW                     ',' 000582 ',' short  ',' 001 ',' decoupler BW                        ';
%
   'Acquisition.Grd_Orient     ',' 000584 ',' char   ',' 004 ',' gradient orientation                ';
%    'LatchLP                    ',' 000588 ',' long   ',' 001 ',' 990629JMB values for latched LP brd ';
   'Acquisition.Grd_Theta      ',' 000592 ',' float64',' 001 ',' 990720JMB grad rotation angle theta ';
   'Acquisition.Grd_Phi        ',' 000600 ',' float64',' 001 ',' 990720JMB grad rotation angle phi   ';
%   'spaces6                    ',' 000608 ',' char   ',' 264 ','                                     ';
%
% time variables (CTime is number of seconds since 0:00:00 Jan 1, 1970;
% elapsed time is in seconds as well)
   'Misc.start_time            ',' 000872 ',' long   ',' 001 ',' starting time                       ';
   'Misc.finish_time           ',' 000876 ',' long   ',' 001 ',' finishing time                      ';
   'Misc.elapsed_time          ',' 000880 ',' long   ',' 001 ',' projected elapsed time (s)          ';

% text variables
   'Misc.Date                  ',' 000884 ',' char   ',' 032 ',' Experiment date                     ';
   'Acquisition.Nucleus        ',' 000916 ',' char   ',' 016 ',' nucleus                             '];
%    'nucleus_2D                 ',' 000932 ',' char   ',' 016 ',' 2D nucleus                          ';
%    'nucleus_3D                 ',' 000948 ',' char   ',' 016 ',' 3D nucleus                          ';
%    'nucleus_4D                 ',' 000964 ',' char   ',' 016 ',' 4D nucleus                          ';
%    'sequence                   ',' 000980 ',' char   ',' 032 ',' sequence name                       ';
%    'lock_solvent               ',' 001012 ',' char   ',' 016 ',' Lock solvent                        ';
%    'lock_nucleus               ',' 001028 ',' char   ',' 016 ',' Lock nucleus                        '];
end % Function Read_TNT_hcr

function M = Read_TMG2()
M=[% Display Menu flags:
   'Disp_menu.real_flag        ',' 000000 ',' ubit4  ',' 001 ',' display real data                   ';
   'Disp_menu.imag_flag        ',' 000004 ',' ubit4  ',' 001 ',' display imaginary data              ';
   'Disp_menu.magn_flag        ',' 000008 ',' ubit4  ',' 001 ',' display magnitude data              ';
   'Disp_menu.axis_visible     ',' 000012 ',' ubit4  ',' 001 ',' display axis                        ';
   'Disp_menu.auto_scale       ',' 000016 ',' ubit4  ',' 001 ',' auto scale mode on or off           ';
   'Disp_menu.line_display     ',' 000020 ',' ubit4  ',' 001 ',' TRUE for lines, FALSE for points    ';
   'Disp_menu.show_shim_units  ',' 000024 ',' ubit4  ',' 001 ',' display shim units on the data area ';
   
% Option Menu flags:
   'Opts_menu.integral_display ',' 000028 ',' ubit4  ',' 001 ',' integrals on? - but not swap area   ';
   'Opts_menu.fit_display      ',' 000032 ',' ubit4  ',' 001 ',' fits turned on?  - but not swap area';
   'Opts_menu.show_pivot       ',' 000036 ',' ubit4  ',' 001 ',' show pivot point (inter. phasing)   ';
   'Opts_menu.label_peaks      ',' 000040 ',' ubit4  ',' 001 ',' show labels on the peaks?           ';
   'Opts_menu.keep_manual_peaks',' 000044 ',' ubit4  ',' 001 ',' keep man pks                        '; %when re-applying pk pick settings?
   'Opts_menu.label_peaks      ',' 000048 ',' ubit4  ',' 001 ',' peak label type                     ';
   'Opts_menu.intgrl_dc_avg    ',' 000052 ',' ubit4  ',' 001 ',' use dc avg for integral calculation ';
   'Opts_menu.intgrl_show_mltpr',' 000056 ',' ubit4  ',' 001 ',' show multiplier on scaled integrals ';
   'Opts_menu.Boolean_space    ',' 000060 ',' ubit4  ',' 009 ','                                     ';

% Processing flags:
   'Processing.all_ffts_done   ',' 000096 ',' ubit4  ',' 004 ','                                     ';
   'Processing.all_phase_done  ',' 000112 ',' ubit4  ',' 004 ','                                     ';

% Vertical display multipliers:
   'amp                        ',' 000128 ',' float64',' 001 ',' amplitude scale factor              ';
   'ampbits                    ',' 000136 ',' float64',' 001 ',' resolution of display               ';
   'ampCtl                     ',' 000144 ',' float64',' 001 ',' amplitude control value             ';
   'offset                     ',' 000152 ',' long   ',' 001 ',' vertical offset                     ';

% grid_and_axis	axis_set;					256		see Grid and Axis Structure below
% Grid and Axis Structure
   'Grid_Axis.majorTickInc     ',' 000156 ',' float64',' 012 ',' Increment between major ticks       ';
   'Grid_Axis.minorIntNum      ',' 000252 ',' short  ',' 012 ',' Num of intervals between major ticks'; % (minor ticks is one less than this)
   'Grid_Axis.labelPrecision   ',' 000276 ',' short  ',' 012 ',' Num of digits after decimal point   ';
   'Grid_Axis.gaussPerCentimetr',' 000300 ',' float64',' 001 ',' for calc of distance in freq domain ';
   'Grid_Axis.gridLines        ',' 000308 ',' short  ',' 001 ',' Num horizontal grid lines shown     ';
   'Grid_Axis.axisUnits        ',' 000310 ',' short  ',' 001 ',' Type of units to show               '; % see constants.h

   'Grid_Axis.showGrid         ',' 000312 ',' ubit4  ',' 004 ',' Show or hide the grid               ';
   'Grid_Axis.showGridLabels   ',' 000316 ',' ubit4  ',' 001 ',' Show/hide labels on the grid lines  ';

   'Grid_Axis.adjustOnZoom     ',' 000320 ',' ubit4  ',' 001 ',' Adjust num of ticks and precision   '; % when zoomed in
   'Grid_Axis.showDistanceUnits',' 000324 ',' ubit4  ',' 001 ',' show frequency or distance units    '; % when in frequency domain
   'Grid_Axis.axisName         ',' 000328 ',' char   ',' 032 ',' file name of the axis               '; % (not used as of 4/10/97)

   'space                      ',' 000360 ',' char   ',' 052 ','                                     ';


   'display_units              ',' 000412 ',' short  ',' 004 ',' display units for swap area         ';
   'ref_point                  ',' 000420 ',' long   ',' 004 ',' used in freq offset calcs           ';
   'ref_value                  ',' 000436 ',' float64',' 004 ',' for use in frequency offset calcs   ';
   'z_start                    ',' 000468 ',' long   ',' 001 ',' beginning of data display           '; % (range: 0 to 2 * npts[0] - 2)
   'z_end                      ',' 000472 ',' long   ',' 001 ',' end of data display                 '; % (range: 0 to 2 * npts[0] - 2)
   'z_select_start             ',' 000476 ',' long   ',' 001 ',' beginning of zoom highlight         ';
   'z_select_end               ',' 000480 ',' long   ',' 001 ',' end of zoom highlight               ';
   'last_zoom_start            ',' 000484 ',' long   ',' 001 ',' last z_select_start                 '; % not used yet (4/10/97)
   'last_zoom_end              ',' 000488 ',' long   ',' 001 ',' last z_select_end                   '; % not used yet (4/10/97)
   'index_2D                   ',' 000492 ',' long   ',' 001 ',' in 1D window, which 2D record we see';
   'index_3D                   ',' 000496 ',' long   ',' 001 ',' in 1D window, which 3D record we see';
   'index_4D                   ',' 000500 ',' long   ',' 001 ',' in 1D window, which 4D record we see';
											
	
   'apodization_done           ',' 000504 ',' long   ',' 004 ',' mskd val showing which processing   '; % has been done to the data; see constants.h for values
   'linebrd                    ',' 000520 ',' float64',' 004 ',' line broadening value               ';
   'gaussbrd                   ',' 000552 ',' float64',' 004 ',' gaussian broadening value           ';
   'dmbrd                      ',' 000584 ',' float64',' 004 ',' double exponential broadening value ';
   'sine_bell_shift            ',' 000616 ',' float64',' 004 ',' sine bell shift value               ';
   'sine_bell_width            ',' 000648 ',' float64',' 004 ',' sine bell width value               ';
   'sine_bell_skew             ',' 000680 ',' float64',' 004 ',' sine bell skew value                ';
   'Trapz_point_1              ',' 000712 ',' long   ',' 004 ',' first trpzd pnt for trpzdl apod.    ';
   'Trapz_point_2              ',' 000728 ',' long   ',' 004 ',' second trpzd pnt for trpzdl apod.   ';
   'Trapz_point_3              ',' 000744 ',' long   ',' 004 ',' third trpzd pnt for trpzdl apod     ';
   'Trapz_point_4              ',' 000760 ',' long   ',' 004 ',' fourth trpzd pnt for trpzdl apod.   ';
   'trafbrd                    ',' 000776 ',' float64',' 004 ',' Traficante-Ziessow broadening value ';
   'echo_center                ',' 000808 ',' long   ',' 004 ',' echo center for all dimensions      ';

							

   'data_shift_points          ',' 000824 ',' long   ',' 001 ',' num pnts to use in lt/rt shift opers';
   'fft_flag                   ',' 000828 ',' short  ',' 004 ',' fourier transform done?             '; %false if time domain, true if frequency domain
   'unused                     ',' 000836 ',' float64',' 008 ','                                     ';
   'pivot_point                ',' 000900 ',' long   ',' 004 ',' for interactive phasing             ';
   'cumm_0_phase               ',' 000916 ',' float64',' 004 ',' cumm zero order phase applied       ';
   'cumm_1_phase               ',' 000948 ',' float64',' 004 ',' cumm first order phase applied      ';
   'manual_0_phase             ',' 000980 ',' float64',' 001 ',' used for interactive phasing        ';
   'manual_1_phase             ',' 000988 ',' float64',' 001 ',' used for interactive phasing        ';
   'phase_0_value              ',' 000996 ',' float64',' 001 ',' last zero order phase value applied ';
   'phase_1_value              ',' 001004 ',' float64',' 001 ',' last first order phase value applied';
   'session_phase_0            ',' 001012 ',' float64',' 001 ',' used during interactive phasing     ';
   'session_phase_1            ',' 001020 ',' float64',' 001 ',' used during interactive phasing     ';
											

   'max_index                  ',' 001028 ',' long   ',' 001 ',' index of max data value             ';
   'min_index                  ',' 001032 ',' long   ',' 001 ',' index of min data value             ';
   'peak_threshold             ',' 001036 ',' long   ',' 001 ',' threshold above which pks are chosen';
   'peak_noise                 ',' 001040 ',' long   ',' 001 ',' min value to distinguish two peaks  '; 
   'integral_dc_points         ',' 001044 ',' short  ',' 001 ',' num points in intgrl calc - dc avg  ';
   'integral_label_type        ',' 001046 ',' short  ',' 001 ',' how to label integrals              '; % see constants.h
   'integral_scale_factor      ',' 001048 ',' long   ',' 001 ',' scale factor used in integral draw  ';
   'auto_integrate_shoulder    ',' 001052 ',' long   ',' 001 ',' num pnts determine integral cut off ';
   'auto_integrate_noise       ',' 001056 ',' float64',' 001 ',' min shoulder points avg             ';
   'auto_integrate_threshold   ',' 001064 ',' float64',' 001 ',' pk threshold for auto integrate     ';
   's_n_peak                   ',' 001072 ',' long   ',' 001 ',' pk to be used for SNR calculation   ';
   's_n_noise_start            ',' 001076 ',' long   ',' 001 ',' start of noise region for SNR calc  ';
   's_n_noise_end              ',' 001080 ',' long   ',' 001 ',' end of noise region for SNR calc    ';
   's_n_calculated             ',' 001084 ',' long   ',' 001 ',' calculated signal to noise value    ';


   'Spline_point               ',' 001088 ',' long   ',' 014 ',' pnts used for spline bsln fix calc  ';
   'Spline_point_avr           ',' 001144 ',' short  ',' 001 ',' for baseline fix                    ';
   'Poly_point                 ',' 001146 ',' long   ',' 008 ',' pnts for polynomial bsln fix calc   ';
   'Poly_point_avr             ',' 001178 ',' short  ',' 001 ',' for baseline fix                    ';
   'Poly_order                 ',' 001180 ',' short  ',' 001 ',' what order polynomial to use        ';

% Blank Space:
   'space                      ',' 001182 ',' char   ',' 610 ','                                     ';
														
% Text variables:
   'line_simulation_name       ',' 001792 ',' char   ',' 032 ','                                     ';
   'integral_template_name     ',' 001824 ',' char   ',' 032 ','                                     ';
   'baseline_template_name     ',' 001856 ',' char   ',' 032 ','                                     ';
   'layout_name                ',' 001888 ',' char   ',' 032 ','                                     ';
   'relax_information_name     ',' 001920 ',' char   ',' 032 ','                                     ';

   'username                   ',' 001952 ',' char   ',' 032 ','                                     ';
	
   'user_string_1              ',' 001984 ',' char   ',' 016 ','                                     ';
   'user_string_2              ',' 002000 ',' char   ',' 016 ','                                     ';
   'user_string_3              ',' 002016 ',' char   ',' 016 ','                                     ';
   'user_string_4              ',' 002032 ',' char   ',' 016 ','                                     '];
end