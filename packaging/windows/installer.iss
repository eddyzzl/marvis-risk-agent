#ifndef AppVersion
#define AppVersion "0.0.0"
#endif

#ifndef PayloadDir
#define PayloadDir "dist\windows\build\payload"
#endif

#ifndef OutputDir
#define OutputDir "dist\windows"
#endif

[Setup]
AppId={{5C90AA61-5E51-4C89-8C67-B7EBF02224D2}
AppName=MARVIS-Agent
AppVersion={#AppVersion}
AppPublisher=MARVIS-Agent
AppPublisherURL=https://github.com/eddyzzl/marvis-risk-agent
AppSupportURL=https://github.com/eddyzzl/marvis-risk-agent
AppUpdatesURL=https://github.com/eddyzzl/marvis-risk-agent/releases
DefaultDirName={localappdata}\Programs\MARVIS-Agent
DefaultGroupName=MARVIS-Agent
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=MARVIS-Setup-{#AppVersion}-win-x64
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=MARVIS-Agent
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: checkedonce

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\MARVIS-Agent"; Filename: "{app}\MARVIS-Agent.cmd"; WorkingDir: "{app}"
Name: "{autodesktop}\MARVIS-Agent"; Filename: "{app}\MARVIS-Agent.cmd"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\MARVIS-Agent.cmd"; Description: "Launch MARVIS-Agent"; Flags: nowait postinstall skipifsilent unchecked
