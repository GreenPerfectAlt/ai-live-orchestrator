
const $ = (id) => document.getElementById(id);
const stateColors = {
  loading: ['#3a3d46','rgba(58,61,70,0.14)','Loading...'],
  listening: ['#4ade80','rgba(74,222,128,0.13)','Listening'],
  processing: ['#f59e0b','rgba(245,158,11,0.13)','Thinking...'],
  speaking: ['#818cf8','rgba(129,140,248,0.13)','Speaking'],
  recording: ['#fb7185','rgba(251,113,133,0.14)','Recording...']
};
let ws, state = 'loading', mediaStream = null, screenStream = null, audioCtx = null, analyser = null;
let cameraEnabled = true, screenSending = true, pdfSending = true, videoSending = true, imageSending = true;
let pdfDoc = null, pdfPage = 1, pdfPageCount = 0, pdfRenderTask = null, videoObjectUrl = null;
let uploadedImages = [], imageObjectUrls = [], currentImageIndex = 0;
let messages = [], msgCounter = 0, activeAssistantId = null, activeRequestId = null;
let streamSampleRate = 24000, streamNextTime = 0, streamSources = [];
let recording = false, recSource = null, recProcessor = null, recZeroGain = null, recChunks = [], recSampleRate = 48000, recRawActive = false;
let speechRec = null, speechTranscript = '', speechRunning = false;
let currentAudioMessageId = null, currentAudioRequestId = null, replayTargets = new Map(), messageAudioCache = new Map(), ignoredAudioRequests = new Set(), ignoredTextRequests = new Set();
let ttsPendingAudioChunks = [], ttsPlaybackStarted = false, ttsAudioEnded = false, ttsBufferTimer = null;
let lastAssistantTextForEcho = '', ttsEchoActive = false;
let assistantOutputActive = false, assistantOutputResumeTimer = null, assistantSpokenTexts = [];
let assistantMuteWatchdogTimer = null, assistantMuteToken = 0;
let browserTtsBuffer = '', browserTtsQueue = [], browserTtsSpeaking = false;
let autoSuppressUntil = 0, lastAutoSttSent = '', lastAutoSttAt = 0;
let autoSttDraft = '', autoSttLastAt = 0, autoSttTimer = null, autoSttRestartTimer = null, autoSttSending = false, autoSttSendingAt = 0;
let autoMicWatchdogTimer = null, speechLastStartAt = 0, speechLastResultAt = 0, speechStarting = false;
let autoSttOnlyMode = false;
let micHoldActive = false, micPointerId = null, micStartedAt = 0, micFinalizing = false, micFinalizeTimer = null;
let lastBargeCandidate = '', lastBargeCandidateAt = 0, bargeCandidateHits = 0;
let autoMicEnabled = false, autoMonitoring = false, autoRecording = false, autoSource = null, autoProcessor = null, autoZeroGain = null, autoChunks = [], autoSampleRate = 48000, autoLastVoiceAt = 0, autoStartedAt = 0;
let autoNoiseFloor = 0.004, autoHotFrames = 0, autoPreRoll = [];
let speakingStartedAt = 0, lastBargeInterruptAt = 0;
let liveFrameTimer = null, liveFrameSeq = 0;
let liveFrameBuffers = {camera: [], screen: [], video: []};
let splitDragging = false;

const TAB_ID = `tab-${Date.now()}-${Math.random().toString(16).slice(2)}`;
const audioFocusChannel = ('BroadcastChannel' in window) ? new BroadcastChannel('parlor-audio-focus') : null;
function claimAudioFocus(requestId=''){
  try{ audioFocusChannel?.postMessage({type:'takeover', tabId:TAB_ID, requestId:requestId || ''}); }catch{}
  try{ localStorage.setItem('parlor.audio.owner', JSON.stringify({tabId:TAB_ID, requestId:requestId || '', at:Date.now()})); }catch{}
}
audioFocusChannel && (audioFocusChannel.onmessage = (ev)=>{
  const d = ev.data || {};
  if(d.type === 'takeover' && d.tabId && d.tabId !== TAB_ID){
    stopPlayback({clearRequest:true});
  }
});
window.addEventListener('storage', (ev)=>{
  if(ev.key !== 'parlor.audio.owner' || !ev.newValue) return;
  try{
    const d = JSON.parse(ev.newValue);
    if(d.tabId && d.tabId !== TAB_ID) stopPlayback({clearRequest:true});
  }catch{}
});



// ── Persistent chats + settings ──
const SETTINGS_KEY = 'ai-live-orchestrator.settings.v11.dual-backend';
const CHATS_KEY = 'parlor.jarvis.chats.v10';
const ACTIVE_CHAT_KEY = 'parlor.jarvis.activeChat.v10';
const DEFAULT_SETTINGS = {
  wsUrl: '', voice: '', temperature: 1.0, top_p: 1.0, top_k: 0,
  max_output_tokens: 0, historyMessages: 20,
  cameraMax: 288, screenMax: 384, pdfMax: 448, videoMax: 384, imageMax: 512,
  liveFps: 0.2, liveFrames: 1, autoVoice: false, startAutoWithScreen: true, voiceThreshold: 0.04, silenceStopMs: 950, minSpeechMs: 550,
  bargeIn: 0, bargeInGraceMs: 1500, bargeInSensitivity: 1.35,
  attachCameraAlways: 0, visionMode: 'auto',
  browserStt: 1, serverAsr: 1, sttLang: 'ru-RU', sendRawAudio: 0, audioInputMode: 'stt', ttsMode: 'server', browserTtsRate: 1.0,
  ttsEngine: 'supertonic', sileroSpeaker: 'baya', sileroSpeed: 1.0,
};
let appSettings = loadSettings();
if(!localStorage.getItem(SETTINGS_KEY)){ appSettings.sttLang = DEFAULT_SETTINGS.sttLang || 'ru-RU'; saveSettings(); }
let chats = loadChats();
let activeChatId = localStorage.getItem(ACTIVE_CHAT_KEY) || (chats[0] && chats[0].id) || createChatRecord('New chat').id;
let suppressChatSave = false;

function clampNum(v, d, lo, hi){ v = Number(v); if(!Number.isFinite(v)) return d; return Math.max(lo, Math.min(hi, v)); }
function loadSettings(){ try { return {...DEFAULT_SETTINGS, ...(JSON.parse(localStorage.getItem(SETTINGS_KEY)||'{}')||{})}; } catch { return {...DEFAULT_SETTINGS}; } }
function saveSettings(){ localStorage.setItem(SETTINGS_KEY, JSON.stringify(appSettings)); }
function defaultWsUrl(){ return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`; }
function getActiveWsUrl(){ return (appSettings.wsUrl || '').trim() || defaultWsUrl(); }
function createChatRecord(title){ const c={id:'chat-'+Date.now()+'-'+Math.random().toString(16).slice(2), title:title||'New chat', createdAt:Date.now(), updatedAt:Date.now(), systemPrompt:'', messages:[]}; chats.unshift(c); persistChats(); return c; }
function loadChats(){ try { const arr=JSON.parse(localStorage.getItem(CHATS_KEY)||'[]'); if(Array.isArray(arr) && arr.length) return arr; } catch {} return [{id:'chat-default',title:'New chat',createdAt:Date.now(),updatedAt:Date.now(),systemPrompt:'',messages:[]}]; }
function persistChats(){ localStorage.setItem(CHATS_KEY, JSON.stringify(chats.slice(0,60))); }
function currentChat(){ let c=chats.find(x=>x.id===activeChatId); if(!c){ c=createChatRecord('New chat'); activeChatId=c.id; } return c; }
function saveActiveChat(){ if(suppressChatSave) return; const c=currentChat(); c.messages = messages.filter(m=>!m.pending).map(m=>({id:m.id,role:m.role,text:m.text||'',meta:m.meta||''})); c.systemPrompt = $('systemPrompt')?.value || ''; c.updatedAt = Date.now(); persistChats(); renderChatList(); }
function loadActiveChat(){ const c=currentChat(); suppressChatSave = true; messages = (c.messages||[]).map(m=>({...m,pending:false,id:nowId()})); if($('systemPrompt')) $('systemPrompt').value = c.systemPrompt || ''; renderMessages(); suppressChatSave = false; renderChatList(); }
function maybeTitleChat(text){ const c=currentChat(); if(!c || (c.title && c.title !== 'New chat')) return; const t=String(text||'').trim().replace(/\s+/g,' ').slice(0,42); if(t){ c.title=t; persistChats(); renderChatList(); } }
function renderChatList(){ const list=$('chatList'); if(!list) return; list.innerHTML=''; const sorted=[...chats].sort((a,b)=>(b.updatedAt||0)-(a.updatedAt||0)); for(const c of sorted){ const btn=document.createElement('button'); btn.className='chat-item'+(c.id===activeChatId?' active':''); btn.dataset.id=c.id; const count=(c.messages||[]).length; const date=new Date(c.updatedAt||Date.now()).toLocaleString('en-US',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}); btn.innerHTML=`<div class="chat-item-title">${escapeHtml(c.title||'New chat')}</div><div class="chat-item-meta">${count} msg · ${date}</div>`; btn.onclick=()=>{ activeChatId=c.id; localStorage.setItem(ACTIVE_CHAT_KEY, activeChatId); loadActiveChat(); closeChats(); }; list.appendChild(btn); } }
function openChats(){ $('drawerBackdrop')?.classList.add('open'); $('chatDrawer')?.classList.add('open'); renderChatList(); }
function closeChats(){ $('drawerBackdrop')?.classList.remove('open'); $('chatDrawer')?.classList.remove('open'); }
function openSettings(){ fillSettingsForm(); $('settingsBackdrop')?.classList.add('open'); $('settingsModal')?.classList.add('open'); }
function closeSettings(){ $('settingsBackdrop')?.classList.remove('open'); $('settingsModal')?.classList.remove('open'); }
function fillSettingsForm(){
  const map={settingsWs:'wsUrl',settingsVoice:'voice',settingsSttLang:'sttLang',settingsSileroSpeed:'sileroSpeed',settingsTemp:'temperature',settingsTopP:'top_p',settingsTopK:'top_k',settingsMaxTokens:'max_output_tokens',settingsHistoryTurns:'historyMessages',settingsCameraMax:'cameraMax',settingsScreenMax:'screenMax',settingsPdfMax:'pdfMax',settingsVideoMax:'videoMax',settingsImageMax:'imageMax',settingsLiveFps:'liveFps',settingsLiveFrames:'liveFrames',settingsVoiceThreshold:'voiceThreshold',settingsSilenceMs:'silenceStopMs',settingsMinSpeechMs:'minSpeechMs',settingsBargeInSensitivity:'bargeInSensitivity'};
  for(const [id,k] of Object.entries(map)){ const el=$(id); if(el) el.value = appSettings[k] ?? ''; }
  if($('settingsBackend')) $('settingsBackend').value = appSettings.backend || 'llama_cpp';
  if($('settingsAutoVoice')) $('settingsAutoVoice').value = appSettings.autoVoice ? '1' : '0';
  if($('settingsStartAutoWithScreen')) $('settingsStartAutoWithScreen').value = appSettings.startAutoWithScreen ? '1' : '0';
  if($('settingsTtsEngine')) $('settingsTtsEngine').value = appSettings.ttsEngine || 'supertonic';
  if($('settingsSileroSpeaker')) $('settingsSileroSpeaker').value = appSettings.sileroSpeaker || 'baya';
  if($('settingsBrowserStt')) $('settingsBrowserStt').value = String(Number(appSettings.browserStt ?? 1));
  if($('settingsServerAsr')) $('settingsServerAsr').value = String(Number(appSettings.serverAsr ?? 1));
  if($('settingsSendRawAudio')){
    const mode = appSettings.audioInputMode || (Number(appSettings.sendRawAudio || 0) ? 'native' : 'stt');
    $('settingsSendRawAudio').value = ['stt','native','hybrid'].includes(mode) ? (mode === 'hybrid' ? 'stt' : mode) : 'stt';
  }
  if($('settingsAttachCameraAlways')) $('settingsAttachCameraAlways').value = String(Number(appSettings.attachCameraAlways ?? 0));
  if($('settingsVisionMode')) $('settingsVisionMode').value = appSettings.visionMode || 'auto';
  if($('settingsBargeIn')) $('settingsBargeIn').value = String(Number(appSettings.bargeIn ?? 1));
}
function readSettingsForm(){
  appSettings.wsUrl = $('settingsWs')?.value.trim() || '';
  appSettings.backend = $('settingsBackend')?.value || 'llama_cpp';
  appSettings.voice = $('settingsVoice')?.value.trim() || '';
  appSettings.ttsEngine = $('settingsTtsEngine')?.value || 'supertonic';
  appSettings.sileroSpeaker = $('settingsSileroSpeaker')?.value || 'baya';
  appSettings.sileroSpeed = clampNum($('settingsSileroSpeed')?.value, DEFAULT_SETTINGS.sileroSpeed, 0.85, 1.2);
  appSettings.sttLang = $('settingsSttLang')?.value.trim() || 'ru-RU';
  appSettings.browserStt = Number($('settingsBrowserStt')?.value ?? 1);
  appSettings.serverAsr = Number($('settingsServerAsr')?.value ?? 1);
  appSettings.audioInputMode = $('settingsSendRawAudio')?.value || 'stt';
  if(appSettings.audioInputMode === 'hybrid') appSettings.audioInputMode = 'native';
  if(!['stt','native'].includes(appSettings.audioInputMode)) appSettings.audioInputMode = 'stt';
  appSettings.sendRawAudio = appSettings.audioInputMode === 'stt' ? 0 : 1;
  if(appSettings.backend === 'litertlm'){ appSettings.audioInputMode = 'native'; appSettings.sendRawAudio = 1; appSettings.browserStt = 0; }
  if(appSettings.backend === 'llama_native'){ appSettings.audioInputMode = 'native'; appSettings.sendRawAudio = 1; }
  if(appSettings.backend === 'llama_cpp'){ appSettings.audioInputMode = 'stt'; appSettings.sendRawAudio = 0; }
  appSettings.attachCameraAlways = Number($('settingsAttachCameraAlways')?.value ?? 0);
  appSettings.visionMode = $('settingsVisionMode')?.value || 'auto';
  appSettings.temperature = clampNum($('settingsTemp')?.value, DEFAULT_SETTINGS.temperature, 0, 2);
  appSettings.top_p = clampNum($('settingsTopP')?.value, DEFAULT_SETTINGS.top_p, 0, 1);
  appSettings.top_k = clampNum($('settingsTopK')?.value, DEFAULT_SETTINGS.top_k, 0, 256);
  appSettings.max_output_tokens = clampNum($('settingsMaxTokens')?.value, DEFAULT_SETTINGS.max_output_tokens, -1, 32768);
  appSettings.historyMessages = clampNum($('settingsHistoryTurns')?.value, DEFAULT_SETTINGS.historyMessages, 2, 200);
  appSettings.cameraMax = clampNum($('settingsCameraMax')?.value, DEFAULT_SETTINGS.cameraMax, 224, 1024);
  appSettings.screenMax = clampNum($('settingsScreenMax')?.value, DEFAULT_SETTINGS.screenMax, 224, 1024);
  appSettings.pdfMax = clampNum($('settingsPdfMax')?.value, DEFAULT_SETTINGS.pdfMax, 224, 1024);
  appSettings.videoMax = clampNum($('settingsVideoMax')?.value, DEFAULT_SETTINGS.videoMax, 224, 1024);
  appSettings.imageMax = clampNum($('settingsImageMax')?.value, DEFAULT_SETTINGS.imageMax, 224, 1536);
  appSettings.liveFps = clampNum($('settingsLiveFps')?.value, DEFAULT_SETTINGS.liveFps, 0.2, 8);
  appSettings.liveFrames = clampNum($('settingsLiveFrames')?.value, DEFAULT_SETTINGS.liveFrames, 1, 24);
  appSettings.voiceThreshold = clampNum($('settingsVoiceThreshold')?.value, DEFAULT_SETTINGS.voiceThreshold, 0.002, 0.2);
  appSettings.silenceStopMs = clampNum($('settingsSilenceMs')?.value, DEFAULT_SETTINGS.silenceStopMs, 300, 3000);
  appSettings.minSpeechMs = clampNum($('settingsMinSpeechMs')?.value, DEFAULT_SETTINGS.minSpeechMs, 250, 2000);
  appSettings.autoVoice = $('settingsAutoVoice')?.value === '1';
  appSettings.startAutoWithScreen = $('settingsStartAutoWithScreen')?.value !== '0';
  appSettings.bargeIn = Number($('settingsBargeIn')?.value ?? 1);
  appSettings.bargeInSensitivity = clampNum($('settingsBargeInSensitivity')?.value, DEFAULT_SETTINGS.bargeInSensitivity, 1.0, 3.0);
  appSettings.ttsMode = 'server';
  saveSettings();
  restartLiveFrameRecorder();
  setAutoMic(appSettings.autoVoice);
}

function setState(s) {
  state = s;
  const [glow, glowDim, label] = stateColors[s] || stateColors.loading;
  document.documentElement.style.setProperty('--glow', glow);
  document.documentElement.style.setProperty('--glow-dim', glowDim);
  $('stateText').textContent = label;
  for (const el of [$('cameraTile'), $('screenTile'), $('pdfTile'), $('videoTile'), $('imageTile')]) {
    el.classList.remove('loading','listening','processing','speaking','recording');
    el.classList.add(s === 'recording' ? 'listening' : s);
  }
}
function setStatus(cls, text) { const el=$('status'); el.className=`status-pill ${cls}`; el.textContent=text; }
function nowId() { return ++msgCounter; }
function escapeHtml(s){ return String(s||'').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function textOfMessage(m){ return (m.text || '').replace(/<[^>]*>/g,'').trim(); }
function cleanIncomingText(text, {final=false}={}){
  text = String(text || '').replace(/\x00/g,' ');
  text = text
    .replace(/<\|\/?[^>\n]{0,80}?\|>/g,'')
    .replace(/<\/?(?:think|thought|analysis|reasoning)[^>]*>[\s\S]*?<\/(?:think|thought|analysis|reasoning)>/gi,'')
    .replace(/[ \t]{2,}/g,' ');
  return final ? text.trim() : text;
}
function renderMessages(){
  const box = $('messages'); box.innerHTML = '';
  for (const m of messages) {
    const div = document.createElement('div');
    div.className = `msg ${m.role}${m.pending?' pending':''}`;
    div.dataset.id = m.id; if(m.request_id) div.dataset.requestId = m.request_id;
    const body = m.pending ? '<span class="loading-dots"><span></span><span></span><span></span></span>' : escapeHtml(m.text);
    const meta = m.meta ? `<div class="meta">${escapeHtml(m.meta)}</div>` : '';
    div.innerHTML = `<div class="msg-actions"><button class="msg-action edit" title="edit">✎</button><button class="msg-action del" title="delete">×</button></div><div class="msg-body">${body}</div>${meta}`;
    box.appendChild(div);
  }
  $('transcript').scrollTop = $('transcript').scrollHeight;
}
function addMessage(role, text, meta='', pending=false, extra={}){ const m={id:nowId(),role,text,meta,pending,...(extra||{})}; messages.push(m); renderMessages(); return m.id; }
function updateMessage(id, patch){ const m=messages.find(x=>x.id===id); if(m){ Object.assign(m,patch); renderMessages(); } }
function deleteMessage(id){ messages = messages.filter(m => m.id !== id); renderMessages(); }
function editMessage(id){ const m=messages.find(x=>x.id===id); if(!m) return; const next = prompt('Edit message:', m.text || ''); if(next!==null){ m.text = next; m.pending=false; renderMessages(); } }
// v18: unified message action handler is installed later.
function historyForServer(){ const n=Math.max(2, Math.min(200, Number(appSettings.historyMessages||64))); return messages.filter(m=>!m.pending && (m.role==='user'||m.role==='assistant') && textOfMessage(m)).slice(-n).map(m=>({role:m.role,text:textOfMessage(m)})); }


let wsHeartbeatTimer = null;
let wsLastPongAt = 0;
function stopWsHeartbeat(){
  if(wsHeartbeatTimer){ clearInterval(wsHeartbeatTimer); wsHeartbeatTimer = null; }
}
function startWsHeartbeat(){
  stopWsHeartbeat();
  wsLastPongAt = Date.now();
  wsHeartbeatTimer = setInterval(()=>{
    if(!ws || ws.readyState !== WebSocket.OPEN) return;
    try{ ws.send(JSON.stringify({type:'ping', t:Date.now()})); }
    catch(e){ console.warn('WS ping send failed', e); }
    if(Date.now() - wsLastPongAt > 45000){
      console.warn('WS pong timeout, forcing reconnect');
      try{ ws.close(4000, 'pong-timeout'); }catch{}
    }
  }, 15000);
}

function connect(){
  ws = new WebSocket(getActiveWsUrl());
  ws.onopen = () => { console.info('[WS OPEN]', getActiveWsUrl()); setStatus('connected','Connected'); setState('listening'); startWsHeartbeat(); };
  ws.onclose = (ev) => { console.warn('[WS CLOSE]', {code:ev.code, reason:ev.reason, wasClean:ev.wasClean}); stopWsHeartbeat(); setStatus('disconnected','Disconnected'); setTimeout(connect, 2000); };
  ws.onerror = (ev) => { console.warn('[WS ERROR]', ev); setStatus('disconnected','Disconnected'); };
  ws.onmessage = ({data}) => {
    const msg = JSON.parse(data);
    if (msg.type === 'pong') { wsLastPongAt = Date.now(); return; }
    if (msg.type === 'app_status') {
      $('modelLabel').textContent = `${msg.backend || 'backend'} · ${msg.model_label || msg.model || 'model'}`;
      $('launcherLabel').textContent = `bat: ${msg.launcher_name || 'unknown'}`;
      return;
    }
    if (msg.type === 'text' || msg.type === 'text_final') {
      if(ignoredTextRequests.has(msg.request_id)){ if(msg.type === 'text_final') ignoredTextRequests.delete(msg.request_id); return; }
      const txt = cleanIncomingText(msg.text || '', {final:true});
      if (msg.transcription !== undefined) {
        const byReq = msg.request_id ? messages.find(m => m.role === 'user' && m.request_id === msg.request_id) : null;
        const last = byReq || [...messages].reverse().find(m => m.role === 'user' && m.pending);
        if (last) {
          const parts=[];
          if(msg.mode) parts.push(String(msg.mode));
          if(msg.audio_sent) parts.push('audio');
          if(msg.image_sent) parts.push('camera');
          if(msg.transcription_source) parts.push(String(msg.transcription_source));
          updateMessage(last.id,{pending:false,text:msg.transcription || '[no speech recognized]',meta:parts.join(' · ') || last.meta});
        }
      }
      if (activeAssistantId) updateMessage(activeAssistantId,{pending:false,text:txt,meta:`LLM ${msg.llm_time || '?'}s`});
      else activeAssistantId = addMessage('assistant', txt, `LLM ${msg.llm_time || '?'}s`);
      addAssistantEchoText(txt);
      if(txt && ttsMode() === 'server') beginAssistantOutputMute('server_tts_pending');
      if(ttsMode() === 'browser'){
        enqueueBrowserTts('', {force:true});
        const finishedId = activeAssistantId;
        activeAssistantId = null; activeRequestId = null;
        setTimeout(()=>{ if(!browserTtsSpeaking && !browserTtsQueue.length && state==='speaking'){ ttsEchoActive=false; setState('listening'); setStatus('connected','Connected'); } }, 120);
      } else if(ttsMode() === 'off'){
        activeAssistantId = null; activeRequestId = null;
        setState('listening'); setStatus('connected','Connected');
      }
      // Server TTS is normally driven by audio_start/audio_end, but if TTS is
      // skipped/not ready, keep Auto Mic alive after the text answer.
      if(ttsMode() === 'server' && autoMicEnabled){
        setTimeout(()=>{
          // v10 fix: if text arrived but server TTS did not start, release the mic mute properly.
          // In v9 we tried to restart STT while assistantOutputActive was still true,
          // so Auto Mic could look dead or behave inconsistently.
          if(autoMicEnabled && !recording && !micHoldActive && !micFinalizing && !autoRecording && !streamSources.length && !ttsPlaybackStarted && state !== 'speaking'){
            setState('listening');
            clearAutoSttDraft();
            endAssistantOutputMute(700);
          }
        }, 1100);
      }
    } else if (msg.type === 'text_delta') {
      if(ignoredTextRequests.has(msg.request_id)) return;
      const delta = cleanIncomingText(msg.text || '', {final:false});
      if (!activeAssistantId) activeAssistantId = addMessage('assistant','', '', false);
      const m = messages.find(x=>x.id===activeAssistantId);
      if (m) {
        const nextText = cleanIncomingText((m.text || '') + delta, {final:false});
        updateMessage(activeAssistantId,{text:nextText, pending:false});
        lastAssistantTextForEcho = nextText || lastAssistantTextForEcho;
        enqueueBrowserTts(delta, {force:false});
      }
    } else if (msg.type === 'audio_start') {
      if(ignoredAudioRequests.has(msg.request_id)) return;
      const replayTargetId = replayTargets.get(msg.request_id);

      // If audio for an old request arrives after a new request started, drop it.
      if(!replayTargetId && activeRequestId && msg.request_id !== activeRequestId){
        ignoredAudioRequests.add(msg.request_id);
        return;
      }

      const shouldUseServerAudio = ttsMode() === 'server' || !!replayTargetId;
      if(!shouldUseServerAudio){
        ignoredAudioRequests.add(msg.request_id);
        return;
      }

      // Only one server-audio request can own playback. Stop previous request first.
      if(currentAudioRequestId && currentAudioRequestId !== msg.request_id){
        ignoredAudioRequests.add(currentAudioRequestId);
        stopPlayback({clearRequest:true});
      }
      claimAudioFocus(msg.request_id);
      resetTtsPlaybackBuffer();
      streamSampleRate = msg.sample_rate || 24000;
      currentAudioRequestId = msg.request_id;
      currentAudioMessageId = replayTargetId || activeAssistantId;
      ttsEchoActive = true;
      speechTranscript = '';
      setState('speaking'); // reserve mic/listening state while we prebuffer audio
      const m = messages.find(x=>x.id===currentAudioMessageId);
      if (m) {
        lastAssistantTextForEcho = m.text || lastAssistantTextForEcho;
        m.audioChunks = [];
        m.audioSampleRate = streamSampleRate;
        messageAudioCache.set(m.id, {chunks:m.audioChunks, sampleRate:streamSampleRate});
      }
    } else if (msg.type === 'audio_chunk') {
      if(ignoredAudioRequests.has(msg.request_id)) return;
      if(currentAudioRequestId && msg.request_id !== currentAudioRequestId){
        ignoredAudioRequests.add(msg.request_id);
        return;
      }
      if(!currentAudioRequestId && activeRequestId && msg.request_id !== activeRequestId && !replayTargets.has(msg.request_id)){
        ignoredAudioRequests.add(msg.request_id);
        return;
      }
      const targetId = currentAudioMessageId || replayTargets.get(msg.request_id) || activeAssistantId;
      const m = messages.find(x=>x.id===targetId);
      if (m) {
        if(!Array.isArray(m.audioChunks)) m.audioChunks = [];
        m.audioChunks.push(msg.audio);
        m.audioSampleRate = streamSampleRate;
        messageAudioCache.set(m.id, {chunks:m.audioChunks, sampleRate:streamSampleRate});
      }
      pushBufferedTtsChunk(msg.audio);
    } else if (msg.type === 'audio_end') {
      if(ignoredAudioRequests.has(msg.request_id)){
        ignoredAudioRequests.delete(msg.request_id);
        if(currentAudioRequestId === msg.request_id) currentAudioRequestId = null;
        return;
      }
      ttsAudioEnded = true;
      if(ttsPendingAudioChunks.length && !ttsPlaybackStarted) startBufferedTtsPlayback();

      const replayTargetId = replayTargets.get(msg.request_id);
      const targetId = currentAudioMessageId || replayTargetId || activeAssistantId;
      const m = messages.find(x=>x.id===targetId);
      if (m && targetId === activeAssistantId) updateMessage(activeAssistantId,{meta: `${m.meta || ''}${m.meta?' · ':''}TTS ${msg.tts_time || 0}s`});
      else if(m) renderMessages();
      if(replayTargetId) replayTargets.delete(msg.request_id);
      if(!replayTargetId){ activeAssistantId = null; activeRequestId = null; }
      currentAudioMessageId = null;

      const finishWhenAudioDrained = () => {
        if(ttsPendingAudioChunks.length && !ttsPlaybackStarted) startBufferedTtsPlayback();
        if(streamSources.length || ttsPendingAudioChunks.length) { setTimeout(finishWhenAudioDrained, 80); return; }
        clearTtsBufferTimer();
        ttsPlaybackStarted = false;
        ttsAudioEnded = false;
        speechTranscript = '';
        if(currentAudioRequestId === msg.request_id) currentAudioRequestId = null;
        if(state==='speaking' || state==='recording') setState('listening');
        if(autoMicEnabled){
          clearAutoSttDraft();
          autoSttSending=false;
          autoSttSendingAt=0;
          endAssistantOutputMute(2200);
        } else {
          assistantOutputActive=false; ttsEchoActive=false;
          setStatus('connected','Connected');
        }
      };
      finishWhenAudioDrained();
    } else if (msg.type === 'error') {
      if (activeAssistantId) updateMessage(activeAssistantId,{pending:false,text:'[ERROR] '+(msg.message||'unknown')});
      else addMessage('assistant','[ERROR] '+(msg.message||'unknown'));
      setState('listening');
      if(autoMicEnabled){
        autoSttSending=false; autoSttSendingAt=0;
        restartBrowserSttSoon(350);
        setStatus('connected','Auto Mic STT listening');
      } else {
        setStatus('connected','Connected');
      }
    }
  };
}

async function ensureMicStream(){
  const alive = micStream && micStream.getAudioTracks().some(t=>t.readyState === 'live');
  if(alive) return micStream;
  try{
    micStream = await navigator.mediaDevices.getUserMedia({
      video:false,
      audio:{echoCancellation:true, noiseSuppression:true, autoGainControl:true}
    });
    return micStream;
  }catch(e){
    console.warn('microphone failed', e);
    setStatus('error','Mic blocked/unavailable');
    throw e;
  }
}

async function startMedia(){
  // Camera and microphone must be independent. If camera fails, Auto Mic must still work.
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({video:{width:960,height:540,facingMode:'user'}, audio:false});
    $('cameraVideo').srcObject = mediaStream;
  } catch(e){
    console.warn('camera failed; mic can still work', e);
    mediaStream = null;
    cameraEnabled = false;
    $('cameraToggle').textContent = 'No Camera';
    $('cameraToggle').classList.remove('active');
  }
  // Do not pre-open microphone here. Auto Mic/STT or push-to-talk will request it when needed.
}
function ensureAudioCtx(){ if(!audioCtx){ audioCtx = new (window.AudioContext || window.webkitAudioContext)(); analyser = audioCtx.createAnalyser(); analyser.fftSize=256; analyser.smoothingTimeConstant=.75; } if(audioCtx.state==='suspended') audioCtx.resume(); }
function captureFrame(video, maxW=640, quality=.75){ if(!video || !video.videoWidth) return null; const c=document.createElement('canvas'); const scale=Math.min(1,maxW/video.videoWidth); c.width=Math.max(1,Math.round(video.videoWidth*scale)); c.height=Math.max(1,Math.round(video.videoHeight*scale)); c.getContext('2d').drawImage(video,0,0,c.width,c.height); return c.toDataURL('image/jpeg', quality).split(',')[1]; }
function captureCamera(){ return cameraEnabled ? captureFrame($('cameraVideo'), Number(appSettings?.cameraMax||448), .68) : null; }
function captureScreen(){ return screenStream && screenSending ? captureFrame($('screenVideo'), Number(appSettings?.screenMax||512), .68) : null; }
function capturePdf(){ const c=$('pdfCanvas'); if(!pdfDoc || !pdfSending || !c.width) return null; const out=document.createElement('canvas'); const scale=Math.min(1,Number(appSettings?.pdfMax||512)/c.width); out.width=Math.max(1,Math.round(c.width*scale)); out.height=Math.max(1,Math.round(c.height*scale)); out.getContext('2d').drawImage(c,0,0,out.width,out.height); return out.toDataURL('image/jpeg',.82).split(',')[1]; }
function captureVideo(){ const v=$('fileVideo'); return v && v.src && videoSending && !v.paused && !v.ended ? captureFrame(v, Number(appSettings?.videoMax||512), .68) : null; }
function resizeImageBlobFromUrl(url, maxW=640, quality=.82){ return new Promise((resolve)=>{ const img=new Image(); img.onload=()=>{ const c=document.createElement('canvas'); const scale=Math.min(1,maxW/img.naturalWidth); c.width=Math.max(1,Math.round(img.naturalWidth*scale)); c.height=Math.max(1,Math.round(img.naturalHeight*scale)); c.getContext('2d').drawImage(img,0,0,c.width,c.height); resolve(c.toDataURL('image/jpeg',quality).split(',')[1]); }; img.onerror=()=>resolve(null); img.src=url; }); }
function frameHash(blob){ return blob ? `${blob.length}:${blob.slice(0,24)}:${blob.slice(-24)}` : ''; }
function pushLiveFrame(source, blob, kind='live'){
  if(!blob) return;
  const buf = liveFrameBuffers[source] || (liveFrameBuffers[source]=[]);
  const h = frameHash(blob); if(buf.length && buf[buf.length-1].hash === h) return;
  buf.push({source, blob, kind, t: Date.now(), index: ++liveFrameSeq, hash: h});
  const keep = Math.max(4, Math.min(48, Math.ceil(Number(appSettings.liveFrames||8) * 4)));
  while(buf.length > keep) buf.shift();
}
function captureLiveTick(){
  // Performance fix: do NOT capture camera continuously.
  // Camera is captured once at send-time. Continuous JPEG encoding was the main page lag source.
  const scr = captureScreen(); if(scr) pushLiveFrame('screen', scr, 'live-screen');
  const vid = captureVideo(); if(vid) pushLiveFrame('video', vid, 'live-video');
  updateLiveLabels();
}
function restartLiveFrameRecorder(){
  if(liveFrameTimer) clearInterval(liveFrameTimer);
  liveFrameTimer = null;
  // No background camera capture. Only poll when screen/video is actively being shared.
  if(!(screenStream && screenSending) && !($('fileVideo')?.src && videoSending)) { updateLiveLabels(); return; }
  const fps = Math.max(.2, Math.min(2, Number(appSettings.liveFps||0.2)));
  liveFrameTimer = setInterval(captureLiveTick, Math.round(1000/fps));
  setTimeout(captureLiveTick, 80);
}
function takeRecent(source, count){ const buf=liveFrameBuffers[source]||[]; return buf.slice(-Math.max(0,count)).map(({hash,...x})=>x); }
function visualIntent(text=''){
  return /(экран|видишь|видно|посмотри|смотри|покажи|что\s+(?:это|там|тут|здесь)|здесь|тут|это\s+что|на\s+этом|камера|картин|фото|изображ|цвет|pdf|пдф|страниц|видео|screen|camera|image|photo|picture|video|document)/i.test(text || '');
}
function collectFrameSequence(opts={}){
  const maxTotal = Math.max(1, Math.min(16, Number(appSettings.liveFrames||2)));
  const includeScreen = opts.includeScreen === true && !!screenStream && !!screenSending;
  const includeVideo = opts.includeVideo === true && !!$('fileVideo').src && !!videoSending;
  const includePdf = opts.includePdf === true && !!pdfDoc && !!pdfSending;
  const includeCamera = opts.includeCamera === true && !!cameraEnabled;
  if(!includeScreen && !includeVideo && !includePdf && !includeCamera) return [];
  captureLiveTick();
  let frames = [];
  if(includeScreen) frames.push(...takeRecent('screen', maxTotal));
  let remaining = Math.max(0, maxTotal - frames.length);
  if(remaining && includeVideo) frames.push(...takeRecent('video', Math.min(remaining, Math.ceil(maxTotal/2))));
  remaining = Math.max(0, maxTotal - frames.length);
  if(remaining && includeCamera) frames.push(...takeRecent('camera', Math.min(remaining, 1)));
  // If buffer is still empty right after enabling a source, fall back to current frame.
  if(includeScreen && !frames.some(f=>f.source==='screen')){ const scr=captureScreen(); if(scr && frames.length<maxTotal) frames.push({source:'screen', blob:scr, kind:'instant-screen', t:Date.now(), index:++liveFrameSeq}); }
  if(includeVideo && !frames.some(f=>f.source==='video')){ const vid=captureVideo(); if(vid && frames.length<maxTotal) frames.push({source:'video', blob:vid, kind:'instant-video', t:Date.now(), index:++liveFrameSeq}); }
  if(includeCamera && !frames.some(f=>f.source==='camera')){ const cam=captureCamera(); if(cam && frames.length<maxTotal) frames.push({source:'camera', blob:cam, kind:'instant-camera', t:Date.now(), index:++liveFrameSeq}); }
  const pdf=includePdf ? capturePdf() : null; if(pdf && frames.length < Math.min(16, maxTotal+1)) frames.push({source:'pdf', blob:pdf, kind:'current-pdf-page', t:Date.now(), index:++liveFrameSeq});
  frames.sort((a,b)=>(a.t||0)-(b.t||0));
  return frames.slice(-Math.min(16, maxTotal + (includePdf ? 1 : 0)));
}
function collectImages(){ return collectFrameSequence({includeCamera:true}); }
function collectUploadedImages(){
  if(!imageSending || !uploadedImages.length) return [];
  const max = Math.min(8, uploadedImages.length);
  return uploadedImages.slice(0, max).map((blob, i)=>({source:'image', blob, kind:'uploaded-image', index:i+1, t:Date.now()}));
}

function updateLiveLabels(){
  const pairs=[['cameraLiveLabel', cameraEnabled && (liveFrameBuffers.camera||[]).length], ['screenLiveLabel', screenStream && screenSending && (liveFrameBuffers.screen||[]).length], ['videoLiveLabel', $('fileVideo')?.src && videoSending && (liveFrameBuffers.video||[]).length], ['imageLiveLabel', uploadedImages.length && imageSending], ['pdfLiveLabel', pdfDoc && pdfSending]];
  for(const [id,on] of pairs){ const el=$(id); if(el) el.classList.toggle('on', !!on); }
}
var sendPayload = null;

$('sendBtn').onclick = () => { const v=$('textInput').value.trim(); if(v){ $('textInput').value=''; sendPayload({text:v}); } };
$('textInput').addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); $('sendBtn').click(); }});
$('resetBtn').onclick = () => { messages=[]; renderMessages(); activeAssistantId=null; if(ws&&ws.readyState===1) ws.send(JSON.stringify({type:'reset'})); stopPlayback(); setState('listening'); };

function resampleFloat32(input, fromRate, toRate=16000){ if(fromRate===toRate) return input; const ratio=fromRate/toRate; const outLen=Math.round(input.length/ratio); const out=new Float32Array(outLen); for(let i=0;i<outLen;i++){ const pos=i*ratio; const i0=Math.floor(pos); const i1=Math.min(i0+1,input.length-1); const frac=pos-i0; out[i]=input[i0]*(1-frac)+input[i1]*frac; } return out; }

function estimateWavSecondsFromBase64(b64){
  try{
    const byteLen = Math.floor(String(b64 || '').length * 3 / 4);
    const byteRate = 32000; // 16 kHz * 16-bit mono
    return Math.max(0, Math.round(((byteLen - 44) / byteRate) * 10) / 10);
  }catch(e){ return 0; }
}

function float32ToWavBase64(samples){ const buf=new ArrayBuffer(44+samples.length*2); const v=new DataView(buf); const w=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));}; w(0,'RIFF'); v.setUint32(4,36+samples.length*2,true); w(8,'WAVE'); w(12,'fmt '); v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true); v.setUint32(24,16000,true); v.setUint32(28,32000,true); v.setUint16(32,2,true); v.setUint16(34,16,true); w(36,'data'); v.setUint32(40,samples.length*2,true); for(let i=0;i<samples.length;i++){ const s=Math.max(-1,Math.min(1,samples[i])); v.setInt16(44+i*2, s<0?s*0x8000:s*0x7fff, true); } const bytes=new Uint8Array(buf); let bin=''; for(let b of bytes) bin+=String.fromCharCode(b); return btoa(bin); }


function addAssistantEchoText(text=''){
  const t = String(text || '').trim();
  if(!t) return;
  lastAssistantTextForEcho = t;
  assistantSpokenTexts.push(t);
  if(assistantSpokenTexts.length > 8) assistantSpokenTexts = assistantSpokenTexts.slice(-8);
}
function stopBrowserSttHard(){
  try{ if(speechRec) speechRec.abort(); }catch{}
  try{ if(speechRec) speechRec.stop(); }catch{}
  speechRec = null;
  speechRunning = false;
  speechStarting = false;
}
function beginAssistantOutputMute(reason='assistant_output', watchdogMs=18000){
  // v10 fix: mute mic/STT during assistant output, but never leave it muted forever.
  assistantMuteToken += 1;
  const token = assistantMuteToken;
  assistantOutputActive = true;
  ttsEchoActive = true;
  autoSuppressUntil = Math.max(autoSuppressUntil || 0, Date.now() + watchdogMs);
  clearAutoSttDraft();
  stopBrowserSttHard();
  autoRecording = false;
  autoChunks = [];
  autoHotFrames = 0;
  autoPreRoll = [];
  autoSttSending = false;
  autoSttSendingAt = 0;
  if(assistantOutputResumeTimer){ clearTimeout(assistantOutputResumeTimer); assistantOutputResumeTimer = null; }
  if(assistantMuteWatchdogTimer){ clearTimeout(assistantMuteWatchdogTimer); assistantMuteWatchdogTimer = null; }
  assistantMuteWatchdogTimer = setTimeout(()=>{
    // Safety: if server TTS fails to send audio_end, do not keep Auto Mic dead.
    if(token !== assistantMuteToken) return;
    if(assistantOutputActive && !streamSources.length && !ttsPendingAudioChunks.length && !ttsPlaybackStarted){
      console.warn('[v10] Assistant mute watchdog released input after missing/finished TTS', reason);
      endAssistantOutputMute(700);
    }
  }, watchdogMs);
  if(autoMicEnabled && state !== 'speaking') setStatus('connected', 'Assistant speaking · mic paused');
}
function endAssistantOutputMute(delay=1800){
  assistantMuteToken += 1;
  const token = assistantMuteToken;
  if(assistantOutputResumeTimer){ clearTimeout(assistantOutputResumeTimer); assistantOutputResumeTimer = null; }
  if(assistantMuteWatchdogTimer){ clearTimeout(assistantMuteWatchdogTimer); assistantMuteWatchdogTimer = null; }
  autoSuppressUntil = Math.max(autoSuppressUntil || 0, Date.now() + delay);
  assistantOutputResumeTimer = setTimeout(()=>{
    if(token !== assistantMuteToken) return;
    assistantOutputActive = false;
    ttsEchoActive = false;
    clearAutoSttDraft({toListening:true});
    stopBrowserSttHard();
    if(autoMicEnabled && !recording && !micHoldActive && !micFinalizing && !autoRecording && state !== 'processing' && state !== 'speaking'){
      restartBrowserSttSoon(220);
      setStatus('connected','Auto Mic STT listening');
    } else if(!autoMicEnabled && state !== 'processing' && state !== 'speaking'){
      setStatus('connected','Connected');
    }
  }, delay);
}
function isAutoMicMutedByAssistant(){
  return assistantOutputActive || Date.now() < (autoSuppressUntil || 0);
}

function normalizeSpeechText(s=''){
  return String(s || '').toLowerCase()
    .replace(/[ё]/g,'е')
    .replace(/[^\p{L}\p{N}\s]+/gu,' ')
    .replace(/\s+/g,' ')
    .trim();
}
function currentAssistantText(){
  const m = activeAssistantId ? messages.find(x=>x.id===activeAssistantId) : null;
  const pieces = [];
  if(m?.text) pieces.push(m.text);
  if(lastAssistantTextForEcho) pieces.push(lastAssistantTextForEcho);
  for(const t of assistantSpokenTexts) pieces.push(t);
  return pieces.join(' ');
}
function wordOverlapRatio(a='', b=''){
  const aw = normalizeSpeechText(a).split(' ').filter(w=>w.length > 2);
  const bw = new Set(normalizeSpeechText(b).split(' ').filter(w=>w.length > 2));
  if(!aw.length || !bw.size) return 0;
  let hits = 0;
  for(const w of aw) if(bw.has(w)) hits++;
  return hits / aw.length;
}
function isLikelyAssistantEcho(heard=''){
  const h = normalizeSpeechText(heard);
  if(!h) return true;
  const assistant = normalizeSpeechText(currentAssistantText()).slice(-5000);
  if(!assistant) return false;

  if(assistant.includes(h) && h.length > 8) return true;
  if(h.includes(assistant.slice(0, Math.min(120, assistant.length))) && assistant.length > 40) return true;

  const hw = h.split(' ').filter(w=>w.length > 2);
  if(!hw.length) return true;
  const ratio = wordOverlapRatio(h, assistant);

  // Server/browser TTS is often captured with small changes in endings and punctuation.
  // A low threshold is intentional: during Auto Mic we prefer dropping echo over
  // accidentally sending the assistant's own speech as a new user question.
  if(hw.length >= 2 && ratio >= 0.45) return true;
  if(ttsEchoActive && hw.length >= 1 && ratio >= 0.34) return true;
  return false;
}
function shouldBargeFromStt(heard='', isFinal=false){
  if(state !== 'speaking' || !autoMicEnabled) return false;

  const h = normalizeSpeechText(heard);
  if(!h || h.length < 3) return false;

  // Explicit stop words must always stop playback, even if normal barge-in is off.
  const stopLike = /^(стоп|хватит|молчи|замолчи|стой|пауза|перестань|прерви|остановись|нет|stop|wait|pause|shut up)\b/i.test(h);
  if(stopLike) return true;

  if(Number(appSettings.bargeIn ?? 1) !== 1) return false;
  if(Date.now() - speakingStartedAt < Number(appSettings.bargeInGraceMs || 250)) return false;
  if(isLikelyAssistantEcho(h)) return false;

  // Normal interruption: require final STT and at least two words to avoid echo.
  if(!isFinal) return false;
  return h.split(' ').filter(Boolean).length >= 2;
}


function ttsMode(){ return 'server'; }
function cleanForBrowserTts(s=''){
  return String(s || '')
    .replace(/[🌀-🫿✀-➿]+/gu, '')
    .replace(/[*_`#>]+/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}
function browserTtsSupported(){ return 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window; }
function pickBrowserVoice(){
  const voices = speechSynthesis.getVoices ? speechSynthesis.getVoices() : [];
  const lang = (appSettings.sttLang || 'ru-RU').slice(0,2).toLowerCase();
  return voices.find(v => (v.lang || '').toLowerCase().startsWith(lang)) || voices.find(v => /ru|russian/i.test((v.lang||'') + ' ' + (v.name||''))) || voices[0] || null;
}
function extractBrowserTtsChunks(force=false){
  const chunks = [];
  let b = browserTtsBuffer.replace(/\s+/g,' ').trim();
  while(b){
    let split = -1;
    const m = b.match(/^(.{12,220}?[.!?…])(?:\s+|$)/u);
    if(m) split = m[1].length;
    if(split < 0 && (force || b.length >= 180)){
      const cut = Math.min(b.length, 220);
      const sp = b.lastIndexOf(' ', cut);
      split = sp > 60 ? sp : cut;
    }
    if(split < 0) break;
    const part = cleanForBrowserTts(b.slice(0, split));
    if(part) chunks.push(part);
    b = b.slice(split).trim();
  }
  browserTtsBuffer = b;
  return chunks;
}
function enqueueBrowserTts(text, {force=false}={}){
  if(ttsMode() !== 'browser' || !browserTtsSupported()) return;
  if(text) browserTtsBuffer += text;
  const chunks = extractBrowserTtsChunks(force);
  if(chunks.length){
    browserTtsQueue.push(...chunks);
    pumpBrowserTts();
  }
}
function pumpBrowserTts(){
  if(ttsMode() !== 'browser' || !browserTtsSupported()) return;
  if(browserTtsSpeaking || !browserTtsQueue.length) return;
  const text = browserTtsQueue.shift();
  if(!text) return;
  const u = new SpeechSynthesisUtterance(text);
  const v = pickBrowserVoice();
  if(v) u.voice = v;
  u.lang = appSettings.sttLang || 'ru-RU';
  u.rate = Number(appSettings.browserTtsRate || 1.0);
  u.pitch = 1.0;
  u.volume = 1.0;
  browserTtsSpeaking = true;
  beginAssistantOutputMute('browser_tts');
  setState('speaking');
  u.onend = u.onerror = () => {
    browserTtsSpeaking = false;
    if(browserTtsQueue.length) {
      // Small gap prevents Chrome from swallowing the next utterance,
      // but is far shorter than server-side TTS generation gaps.
      setTimeout(pumpBrowserTts, 35);
      return;
    }
    if(!activeAssistantId){
      endAssistantOutputMute(1900);
      speechTranscript = '';
      if(state === 'speaking') setState('listening');
    }
  };
  try{ speechSynthesis.speak(u); }catch(e){ browserTtsSpeaking=false; console.warn('Browser TTS failed', e); }
}
function stopBrowserTts(){
  browserTtsBuffer = '';
  browserTtsQueue = [];
  browserTtsSpeaking = false;
  try{ if(browserTtsSupported()) speechSynthesis.cancel(); }catch{}
}


function clearAutoSttTimer(){
  if(autoSttTimer){ clearTimeout(autoSttTimer); autoSttTimer = null; }
}
function sleep(ms){ return new Promise(resolve => setTimeout(resolve, ms)); }
function chooseBetterTranscript(a='', b=''){
  const aa = normalizeSpeechText(a);
  const bb = normalizeSpeechText(b);
  if(bb.length > aa.length) return b;
  return a || b || '';
}
function restartBrowserSttSoon(delay=220){
  if(autoSttRestartTimer){ clearTimeout(autoSttRestartTimer); autoSttRestartTimer = null; }
  autoSttRestartTimer = setTimeout(()=>{
    autoSttRestartTimer = null;
    ensureAutoMicListening();
  }, delay);
}
function maybeSendAutoStt({force=false}={}){
  if(!autoMicEnabled || autoSttSending) return;
  if(isAutoMicMutedByAssistant()) { clearAutoSttDraft({toListening:true}); return; }
  if(state === 'processing') return;
  const draft = (autoSttDraft || speechTranscript || '').trim();
  const clean = normalizeSpeechText(draft);

  if(!clean || clean.length < 2){
    if(force) clearAutoSttDraft({toListening:true});
    return;
  }

  if(isLikelyAssistantEcho(clean)){
    // Critical fix: do not leave assistant echo in autoSttDraft.
    // A stuck draft made Auto Mic look dead after a few assistant replies.
    clearAutoSttDraft({toListening:true});
    restartBrowserSttSoon(180);
    return;
  }

  const now = Date.now();
  const stopLike = /^(стоп|хватит|молчи|замолчи|стой|пауза|перестань|нет|stop|wait|pause)\b/i.test(clean);
  const enoughWords = clean.split(' ').filter(Boolean).length >= 2;
  const enoughTime = now - autoSttLastAt >= Number(appSettings.silenceStopMs || 900);
  if(!force && !stopLike && !enoughWords && !enoughTime) return;

  if(clean === lastAutoSttSent && now - lastAutoSttAt < 2500){
    clearAutoSttDraft({toListening:true});
    return;
  }

  autoSttSending = true;
  autoSttSendingAt = now;
  autoRecording = false;
  autoChunks = [];
  autoHotFrames = 0;
  lastAutoSttSent = clean;
  lastAutoSttAt = now;
  autoSuppressUntil = now + 1800;

  // Stop current STT cleanly before sending, then watchdog/audio_end will restart it.
  try{ if(speechRec && speechRunning) speechRec.stop(); }catch{}
  speechRec = null;
  speechRunning = false;
  speechStarting = false;

  const textToSend = draft;
  clearAutoSttDraft();

  setState('processing');
  sendPayload({text:textToSend, suppressUserEcho:false});

  setTimeout(()=>{
    autoSttSending = false;
    autoSttSendingAt = 0;
    if(autoMicEnabled) restartBrowserSttSoon(350);
  }, 1200);
}
function scheduleAutoSttSend(){
  clearAutoSttTimer();
  autoSttTimer = setTimeout(()=>maybeSendAutoStt({force:true}), Number(appSettings.silenceStopMs || 900));
}


function clearAutoSttDraft({toListening=false}={}){
  autoSttDraft = '';
  speechTranscript = '';
  clearAutoSttTimer();
  if(toListening && state === 'recording' && !recording && !autoRecording) setState('listening');
}
function shouldKeepSttPaused(){
  if(!autoMicEnabled) return true;
  if(!Number(appSettings.browserStt ?? 1)) return true;
  if(recording || micHoldActive || micFinalizing || autoRecording) return true;
  if(isAutoMicMutedByAssistant()) return true;
  if(state === 'processing') return true;
  if(state === 'speaking') return true;
  return false;
}
function ensureAutoMicListening({force=false}={}){
  if(!autoMicEnabled) return;
  if(!Number(appSettings.browserStt ?? 1)) return;
  if(!force && shouldKeepSttPaused()) return;

  if(force){
    try{ if(speechRec) speechRec.abort(); }catch{}
    speechRec = null;
    speechRunning = false;
    speechStarting = false;
  }

  if(!speechRunning && !speechStarting) startBrowserSTT();
}
function startAutoMicWatchdog(){
  if(autoMicWatchdogTimer) return;
  autoMicWatchdogTimer = setInterval(()=>{
    if(!autoMicEnabled) return;
    const now = Date.now();

    // If a previous autosend got stuck, unlock Auto Mic.
    if(autoSttSending && autoSttSendingAt && now - autoSttSendingAt > 6500){
      console.warn('Auto Mic: autoSttSending stuck, resetting');
      autoSttSending = false;
      autoSttSendingAt = 0;
      clearAutoSttDraft({toListening:true});
    }

    // If UI got stuck in Recording without a real mic hold/VAD recording, reset.
    const lastActivity = Math.max(speechLastResultAt || 0, autoSttLastAt || 0, speechLastStartAt || 0, autoStartedAt || 0);
    if(state === 'recording' && !recording && !micHoldActive && !micFinalizing && !autoRecording && lastActivity && now - lastActivity > 4500){
      console.warn('Auto Mic: stale recording state, resetting');
      clearAutoSttDraft({toListening:true});
      setState('listening');
    }

    if(shouldKeepSttPaused()) return;

    // Chrome SpeechRecognition sometimes silently dies or freezes.
    if(speechStarting && speechLastStartAt && now - speechLastStartAt > 4500){
      console.warn('Auto Mic: STT stuck starting, force restart');
      ensureAutoMicListening({force:true});
      return;
    }
    if(speechRunning && !speechLastResultAt && speechLastStartAt && now - speechLastStartAt > 15000){
      console.warn('Auto Mic: STT running without results, force restart');
      ensureAutoMicListening({force:true});
      return;
    }
    if(speechRunning && speechLastResultAt && now - speechLastResultAt > 18000){
      console.warn('Auto Mic: STT stale, force restart');
      ensureAutoMicListening({force:true});
      return;
    }
    if(!speechRunning && !speechStarting){
      ensureAutoMicListening();
    }
  }, 850);
}

function stopAutoMicWatchdog(){
  if(autoMicWatchdogTimer){ clearInterval(autoMicWatchdogTimer); autoMicWatchdogTimer = null; }
}

function browserSttSupported(){ return !!(window.SpeechRecognition || window.webkitSpeechRecognition); }
function startBrowserSTT(){
  if(speechRunning || speechStarting) return;
  if(isAutoMicMutedByAssistant() && !micHoldActive && !recording) return;
  speechTranscript = '';
  if(!Number(appSettings.browserStt ?? 1)) return;
  if(!browserSttSupported()){ console.warn('Browser STT unsupported in this browser'); setStatus('error','Browser STT unsupported'); return; }

  try{ if(speechRec) speechRec.abort(); }catch{}
  speechRec = null;
  speechStarting = true;
  speechLastStartAt = Date.now();

  try{
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    speechRec = new SR();
    speechRec.lang = appSettings.sttLang || navigator.language || 'en-US';
    speechRec.continuous = true;
    speechRec.interimResults = true;
    speechRec.maxAlternatives = 1;

    speechRec.onstart = ()=>{
      speechRunning = true;
      speechStarting = false;
      speechLastStartAt = Date.now();
      speechLastResultAt = 0;
      if(recording) setStatus('connected','Push mic: STT listening');
      else if(autoMicEnabled) setStatus('connected','Auto Mic STT listening');
    };

    speechRec.onresult = (ev)=>{
      speechLastResultAt = Date.now();
      if(isAutoMicMutedByAssistant() && !micHoldActive && !recording){
        speechTranscript = '';
        clearAutoSttDraft();
        return;
      }
      let finalText = '', interim = '';
      for(let i=0; i<ev.results.length; i++){
        const txt = (ev.results[i][0]?.transcript || '').trim();
        if(ev.results[i].isFinal) finalText += txt + ' ';
        else interim += txt + ' ';
      }
      const heard = (finalText || interim).trim();
      speechTranscript = heard;

      const isFinalSpeech = !!finalText.trim();
      const heardClean = normalizeSpeechText(heard);
      if(!heardClean) return;

      // Main Auto Mic path: Browser STT sends text directly.
      if(autoMicEnabled && !micHoldActive && !recording && !micFinalizing && !autoSttSending && (state === 'listening' || state === 'recording') && !isLikelyAssistantEcho(heardClean)){
        autoSttDraft = heard;
        autoSttLastAt = Date.now();
        if(state === 'listening') setState('recording');
        if(isFinalSpeech) maybeSendAutoStt({force:true});
        else scheduleAutoSttSend();
        return;
      }

      // If it was assistant echo while we were in recording state, clear it,
      // otherwise autoSttDraft can block the fallback VAD forever.
      if(autoMicEnabled && !micHoldActive && !recording && !micFinalizing && isLikelyAssistantEcho(heardClean)){
        clearAutoSttDraft({toListening:true});
        return;
      }

      // Optional barge-in while assistant speaks.
      if(!micHoldActive && !recording && !micFinalizing && shouldBargeFromStt(heard, isFinalSpeech)){
        requestInterrupt(true);
        autoHotFrames = 0;
        autoLastVoiceAt = Date.now();
        if(isFinalSpeech && !/^(стоп|хватит|молчи|замолчи|стой|пауза|перестань|stop|wait|pause)$/i.test(heardClean)){
          autoSttDraft = heard;
          setTimeout(()=>maybeSendAutoStt({force:true}), 120);
        }
      }
    };

    speechRec.onerror = (e)=>{
      const err = e.error || e.message || e;
      console.warn('Mic speech recognition error:', err);
      speechRunning = false;
      speechStarting = false;
      // no-speech/audio-capture/network can happen after a few turns. Watchdog restarts it.
      if(recording) setStatus(err === 'not-allowed' ? 'error' : 'connected', err === 'not-allowed' ? 'Mic permission blocked' : 'Push mic: raw fallback');
      if(autoMicEnabled){ setStatus(err === 'not-allowed' ? 'error' : 'connected', err === 'not-allowed' ? 'Mic permission blocked' : 'Auto Mic raw fallback/listening'); restartBrowserSttSoon(err === 'not-allowed' ? 2000 : 450); }
    };

    speechRec.onend = ()=>{
      speechRunning = false;
      speechStarting = false;
      speechRec = null;
      if(autoMicEnabled && !autoSttSending && !recording && !micFinalizing) restartBrowserSttSoon(220);
    };

    speechRec.start();
  }catch(e){
    console.warn('Browser STT start failed:', e?.message || e);
    speechRec = null;
    speechRunning = false;
    speechStarting = false;
    if(autoMicEnabled) restartBrowserSttSoon(900);
  }
}
function stopBrowserSTT(){
  const text = (speechTranscript || '').trim();
  try{ if(speechRec && (speechRunning || speechStarting)) speechRec.stop(); }catch{}
  speechRec = null; speechRunning = false; speechStarting = false;
  return text;
}
async function stopBrowserSTTFinal(delay=320){
  const before = (speechTranscript || '').trim();
  try{ if(speechRec && (speechRunning || speechStarting)) speechRec.stop(); }catch{}
  // Chrome may deliver the final transcript AFTER stop() / button release.
  // Wait a little so the last word is not cut off.
  await sleep(delay);
  const after = (speechTranscript || '').trim();
  speechRec = null; speechRunning = false; speechStarting = false;
  return chooseBetterTranscript(before, after);
}

async function transcribeWavOnServer(wavBase64, {source='voice'}={}){
  if(!Number(appSettings.serverAsr ?? 1) || !wavBase64) return {text:'', ok:false, error:'server_asr_disabled'};
  try{
    setStatus('processing','Transcribing voice...');
    const res = await fetch('/api/asr', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({audio:wavBase64, language: appSettings.sttLang || 'ru-RU', source})
    });
    const data = await res.json().catch(()=>({}));
    if(!res.ok || !data.ok){
      console.warn('Server ASR failed:', data);
      return {text:'', ok:false, error:data.error || ('HTTP '+res.status)};
    }
    return {text:String(data.text || '').trim(), ok:true, latency:data.latency};
  }catch(e){
    console.warn('Server ASR request failed:', e);
    return {text:'', ok:false, error:e?.message || String(e)};
  }
}

async function sendVoiceWavWithFallback(wavBase64, transcript='', {source='voice', allowNative=false}={}){
  let text = normalizeSpeechText(transcript).length ? String(transcript || '').trim() : '';
  if(!text){
    const asr = await transcribeWavOnServer(wavBase64, {source});
    text = String(asr.text || '').trim();
    if(text) console.info('Server ASR transcript:', text);
  }
  if(text){
    if(isLikelyAssistantEcho(text)){
      console.warn('[v10] Dropped likely assistant echo from voice fallback:', text);
      clearAutoSttDraft({toListening:true});
      setState('listening');
      setStatus('connected','Echo ignored · listening');
      if(autoMicEnabled) restartBrowserSttSoon(450);
      return false;
    }
    sendPayload({text});
    return true;
  }
  if(allowNative){
    sendPayload({audio:wavBase64, transcript:''});
    return true;
  }
  addMessage('user','[no speech recognized]', 'mic: no transcript');
  setState('listening');
  setStatus('connected','No speech recognized · check mic/settings');
  if(autoMicEnabled) restartBrowserSttSoon(550);
  return false;
}

async function startRecording(){
  if(recording || micHoldActive || micFinalizing) return;
  micHoldActive = true;
  micStartedAt = Date.now();
  recording = true;
  recRawActive = false;

  // Visual feedback FIRST, before any permission/model/STT async code.
  const btn = $('micBtn');
  btn?.classList.add('recording');
  if(btn) btn.textContent = '🔴';
  setState('recording');
  setStatus('connected','Push mic: starting');

  requestInterrupt(true);
  clearAutoSttDraft();
  recChunks=[];
  speechTranscript='';
  autoSttDraft='';

  // Start browser STT immediately from the user's click gesture.
  // If Chrome STT fails, raw WAV fallback below still records.
  if(Number(appSettings.browserStt ?? 1) === 1 && browserSttSupported()){
    // Important: aborting Auto Mic STT is not enough. Reset flags before starting
    // push-to-talk STT, otherwise startBrowserSTT() returns immediately.
    try{ if(speechRec) speechRec.abort(); }catch{}
    speechRec = null;
    speechRunning = false;
    speechStarting = false;
    startBrowserSTT();
  }

  // Always try raw recording too. This makes the button work even when
  // Chrome SpeechRecognition silently dies / has no result / no internet.
  try{
    await ensureMicStream();
    if(!micHoldActive || !recording) return;
    ensureAudioCtx();
    const tracks = micStream?.getAudioTracks() || [];
    if(!tracks.length) throw new Error('no audio tracks');

    recSampleRate=audioCtx.sampleRate;
    const holdStream = new MediaStream(tracks);
    recSource = audioCtx.createMediaStreamSource(holdStream);
    recProcessor = audioCtx.createScriptProcessor(4096,1,1);
    recZeroGain=audioCtx.createGain();
    recZeroGain.gain.value=0;
    recProcessor.onaudioprocess = (e)=>{
      if(recording) recChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    recSource.connect(recProcessor);
    recProcessor.connect(recZeroGain);
    recZeroGain.connect(audioCtx.destination);
    recRawActive = true;
    setStatus('connected', Number(appSettings.browserStt ?? 1) === 1 ? 'Push mic: STT listening' : 'Push mic: raw recording');
  }catch(e){
    console.warn('Push mic raw fallback unavailable:', e);
    recRawActive = false;
    if(!browserSttSupported() || Number(appSettings.browserStt ?? 1) !== 1){
      setStatus('error','Mic unavailable');
    }else{
      setStatus('connected','Push mic: STT only');
    }
  }
}

function stopRecording(){
  if(micFinalizing) return;
  if(!micHoldActive && !recording) return;

  // Button released: don't send immediately. Chrome STT often finalizes the
  // last word a few hundred ms AFTER release, so keep recording briefly.
  micHoldActive=false;
  micFinalizing=true;

  const btn = $('micBtn');
  btn?.classList.remove('recording');
  if(btn) btn.textContent = '⏳';
  setStatus('connected','Push mic: finishing...');

  if(micFinalizeTimer) clearTimeout(micFinalizeTimer);
  micFinalizeTimer = setTimeout(()=>finalizeRecording(), 520);
}

async function finalizeRecording(){
  if(!micFinalizing && !recording) return;
  micFinalizing=false;
  if(micFinalizeTimer){ clearTimeout(micFinalizeTimer); micFinalizeTimer=null; }

  recording=false;

  const transcriptBeforeStop = (speechTranscript || autoSttDraft || '').trim();
  let transcript = await stopBrowserSTTFinal(360);
  transcript = chooseBetterTranscript(transcriptBeforeStop, transcript).trim();
  if(isLikelyAssistantEcho(transcript)) transcript = '';

  const btn = $('micBtn');
  if(btn) btn.textContent = '🎙';

  try{recProcessor?.disconnect(); recSource?.disconnect(); recZeroGain?.disconnect();}catch{}
  recProcessor=null; recSource=null; recZeroGain=null;

  const len=recChunks.reduce((a,c)=>a+c.length,0);
  const hasEnoughRaw = recRawActive && len >= recSampleRate*0.12;
  const backendMode = appSettings.backend || 'llama_cpp';
  const micModeRaw = appSettings.audioInputMode || (Number(appSettings.sendRawAudio || 0) ? 'native' : 'stt');
  const micMode = backendMode === 'litertlm' ? 'native' : (backendMode === 'llama_native' ? 'native' : (['stt','native','hybrid'].includes(micModeRaw) ? (micModeRaw === 'hybrid' ? 'stt' : micModeRaw) : 'stt'));
  const wantsRaw = micMode === 'native';

  if(transcript && !(wantsRaw && hasEnoughRaw)){
    recChunks=[];
    recRawActive=false;
    clearAutoSttDraft();
    sendPayload({text:transcript});
    if(autoMicEnabled) setTimeout(()=>{ if(autoMicEnabled && !recording && !micHoldActive && !micFinalizing) restartBrowserSttSoon(900); }, 900);
    return;
  }

  if(hasEnoughRaw){
    const merged=new Float32Array(len);
    let off=0;
    for(const c of recChunks){ merged.set(c,off); off+=c.length; }
    recChunks=[];
    recRawActive=false;
    clearAutoSttDraft();
    const wav = float32ToWavBase64(resampleFloat32(merged, recSampleRate, 16000));
    setStatus('processing', wantsRaw ? 'Sending native audio fallback' : 'Sending voice text');
    sendPayload({audio:wav, transcript:transcript});
    if(autoMicEnabled) setTimeout(()=>{ if(autoMicEnabled && !recording && !autoRecording) restartBrowserSttSoon(350); }, 450);
    return;
  }

  recChunks=[];
  recRawActive=false;
  clearAutoSttDraft();
  setState('listening');
  setStatus('connected','No speech recognized');
  if(autoMicEnabled) restartBrowserSttSoon(300);
}

function rmsOf(buf){ let sum=0; for(let i=0;i<buf.length;i++){ const v=buf[i]; sum += v*v; } return Math.sqrt(sum / Math.max(1, buf.length)); }
function updateAutoMicUI(){
  const on = !!autoMicEnabled;
  for(const id of ['autoMicBtn','screenAutoMicBtn']){ const el=$(id); if(el){ el.classList.toggle('live-on', on); el.textContent = on ? 'Auto Mic On' : 'Auto Mic'; } }
  appSettings.autoVoice = on; saveSettings();
}
async function setAutoMic(on){
  autoMicEnabled = !!on; updateAutoMicUI();
  if(autoMicEnabled){
    clearAutoSttDraft();
    autoSttSending=false;
    autoSttSendingAt=0;
    autoSttOnlyMode = false;

    // v8 stable: run Browser STT AND a lightweight raw VAD recorder.
    // If Browser STT hears speech, we send text immediately. If it hears nothing,
    // the raw recording is transcribed by local server ASR. This is slower than
    // Browser STT but actually reliable when Chrome STT silently fails.
    startAutoMicWatchdog();
    setState('listening');
    if(Number(appSettings.browserStt ?? 1)) ensureAutoMicListening({force:true});
    try{ await startAutoMonitor(); }
    catch(e){ console.warn('Auto Mic raw monitor failed:', e); }
    setStatus('connected','Auto Mic listening · STT + ASR fallback');
  } else {
    stopAutoMicWatchdog();
    clearAutoSttDraft();
    autoSttSending=false;
    autoSttSendingAt=0;
    autoSttOnlyMode=false;
    stopAutoMonitor();
    setStatus('connected','Connected');
  }
}
function stopAutoMonitorRawOnly(){
  autoMonitoring = false;
  autoRecording = false;
  autoChunks = [];
  try{autoProcessor?.disconnect(); autoSource?.disconnect(); autoZeroGain?.disconnect();}catch{}
  autoProcessor=null; autoSource=null; autoZeroGain=null;
}

async function startAutoMonitor(){
  if(autoMonitoring) return;

  // Raw VAD fallback when Browser STT is disabled/unavailable.
  // If WebAudio mic fails, keep Auto Mic alive via STT instead of turning it off.
  try{ await ensureMicStream(); }
  catch(e){
    console.warn('Auto Mic raw fallback unavailable:', e);
    if(Number(appSettings.browserStt ?? 1) === 1){
      autoMonitoring = false;
      return;
    }
    autoMicEnabled=false; updateAutoMicUI(); return;
  }
  ensureAudioCtx();
  const tracks = micStream?.getAudioTracks() || [];
  if(!tracks.length){
    console.warn('Auto Mic: microphone unavailable');
    if(Number(appSettings.browserStt ?? 1) === 1){
      autoMonitoring = false;
      return;
    }
    autoMicEnabled=false; updateAutoMicUI(); return;
  }
  const vadStream = new MediaStream(tracks);
  autoSource = audioCtx.createMediaStreamSource(vadStream);
  autoProcessor = audioCtx.createScriptProcessor(2048,1,1);
  autoZeroGain = audioCtx.createGain(); autoZeroGain.gain.value = 0;
  autoSampleRate = audioCtx.sampleRate;
  autoNoiseFloor = 0.006; autoHotFrames = 0; autoPreRoll = [];
  if(Number(appSettings.browserStt ?? 1)) startBrowserSTT();
  autoProcessor.onaudioprocess = (e)=>{
    if(!autoMicEnabled || recording) return;
    if(autoSttDraft) return;
    if(isAutoMicMutedByAssistant()) return;
    if(state === 'processing') return;
    const input = new Float32Array(e.inputBuffer.getChannelData(0));
    const rms = rmsOf(input);
    const now = Date.now();
    const baseThreshold = Number(appSettings.voiceThreshold || 0.04);
    const isSpeakingNow = state === 'speaking';
    const normalThreshold = Math.max(baseThreshold, autoNoiseFloor * 4.0, 0.018);

    if(!autoRecording){
      autoPreRoll.push(input);
      if(autoPreRoll.length > 48) autoPreRoll.shift();

      // Do NOT use raw RMS to interrupt while the assistant speaks.
      // Speakers/echo make RMS unreliable and caused self-interrupts.
      // Smart Browser STT above handles barge-in by ignoring assistant-echo text.
      if(isSpeakingNow){
        autoHotFrames = 0;
        return;
      }

      autoNoiseFloor = autoNoiseFloor * 0.985 + Math.min(rms, baseThreshold) * 0.015;
      if(rms >= normalThreshold) autoHotFrames += 1; else autoHotFrames = Math.max(0, autoHotFrames - 1);

      if(autoHotFrames >= 3){
        autoRecording = true;
        autoStartedAt = now;
        autoLastVoiceAt = now;
        autoChunks = autoPreRoll.slice();
        autoPreRoll = [];
        // STT is already running in Auto Mic mode; do not reset transcript here,
        // otherwise the first word can be cut before VAD confirms speech.
        setState('recording');
      }
      return;
    }

    autoChunks.push(input);
    if(rms >= normalThreshold * 0.75){
      autoLastVoiceAt = now;
    }
    const minSpeechMs = Number(appSettings.minSpeechMs || 550);
    if(now - autoLastVoiceAt > Number(appSettings.silenceStopMs || 950) && now - autoStartedAt >= minSpeechMs){
      finishAutoRecording();
    }
  };
  autoSource.connect(autoProcessor); autoProcessor.connect(autoZeroGain); autoZeroGain.connect(audioCtx.destination);
  autoMonitoring = true;
}
function stopAutoMonitor(){
  autoMicEnabled = false;
  autoMonitoring = false;
  autoRecording = false;
  autoChunks = [];
  try{autoProcessor?.disconnect(); autoSource?.disconnect(); autoZeroGain?.disconnect();}catch{}
  stopBrowserSTT();
  clearAutoSttDraft(); autoSttSending=false; autoSttSendingAt=0;
  autoProcessor=null; autoSource=null; autoZeroGain=null;
  updateAutoMicUI();
  if(state === 'recording') setState('listening');
}
async function finishAutoRecording(){
  if(!autoRecording) return;
  autoRecording = false;
  autoHotFrames = 0;
  const len = autoChunks.reduce((a,c)=>a+c.length,0);
  const minLen = autoSampleRate * (Number(appSettings.minSpeechMs || 550) / 1000);
  if(len < minLen){ autoChunks=[]; stopBrowserSTT(); setState('listening'); if(autoMicEnabled) restartBrowserSttSoon(250); return; }
  const merged = new Float32Array(len); let off=0; for(const c of autoChunks){ merged.set(c,off); off+=c.length; }
  autoChunks=[];
  const wav = float32ToWavBase64(resampleFloat32(merged, autoSampleRate, 16000));
  let transcript = stopBrowserSTT();
  if(isLikelyAssistantEcho(transcript)) transcript = '';
  const allowNative = (appSettings.audioInputMode || 'stt') === 'native';
  await sendVoiceWavWithFallback(wav, transcript, {source:'auto_mic', allowNative});
  if(autoMicEnabled) setTimeout(()=>{ if(autoMicEnabled && !recording && !autoRecording) restartBrowserSttSoon(350); }, 450);
}
async function startMediaIfNeeded(){ if(!mediaStream) await startMedia(); }
const micButton = $('micBtn');

async function beginMicHold(e){
  e?.preventDefault?.();
  e?.stopPropagation?.();
  if(e?.pointerId != null){
    micPointerId = e.pointerId;
    try{ micButton?.setPointerCapture(e.pointerId); }catch{}
  }
  await startRecording();
}
function endMicHold(e){
  e?.preventDefault?.();
  e?.stopPropagation?.();
  if(e?.pointerId != null){
    try{ micButton?.releasePointerCapture(e.pointerId); }catch{}
  }
  micPointerId = null;
  stopRecording();
}

micButton?.addEventListener('pointerdown', beginMicHold);
micButton?.addEventListener('pointerup', endMicHold);
micButton?.addEventListener('pointercancel', endMicHold);

// Fallbacks for browsers/devices where pointer events are weird.
micButton?.addEventListener('mousedown', (e)=>{ if(!micHoldActive) beginMicHold(e); });
window.addEventListener('mouseup', (e)=>{ if(micHoldActive) endMicHold(e); });
micButton?.addEventListener('touchstart', (e)=>{ if(!micHoldActive) beginMicHold(e); }, {passive:false});
window.addEventListener('touchend', (e)=>{ if(micHoldActive) endMicHold(e); }, {passive:false});

window.addEventListener('blur', ()=>{ if(micHoldActive) stopRecording(); });
document.addEventListener('visibilitychange', ()=>{ if(document.hidden && micHoldActive) stopRecording(); });
$('autoMicBtn')?.addEventListener('click', ()=> setAutoMic(!autoMicEnabled));
$('screenAutoMicBtn')?.addEventListener('click', ()=> setAutoMic(!autoMicEnabled));


function clearTtsBufferTimer(){
  if(ttsBufferTimer){ clearTimeout(ttsBufferTimer); ttsBufferTimer = null; }
}
function resetTtsPlaybackBuffer(){
  clearTtsBufferTimer();
  ttsPendingAudioChunks = [];
  ttsPlaybackStarted = false;
  ttsAudioEnded = false;
}
function startBufferedTtsPlayback(){
  if(ttsPlaybackStarted) return;
  claimAudioFocus(currentAudioRequestId || activeRequestId || '');
  if(ttsPlaybackStarted) return;
  clearTtsBufferTimer();
  ensureAudioCtx();
  try{ if(audioCtx && audioCtx.state === 'suspended') audioCtx.resume(); }catch{}
  speakingStartedAt = Date.now();
  beginAssistantOutputMute('server_tts_playback');
  // Slight buffer lead: prevents the first phrase from starting instantly and
  // then falling into silence while Supertonic is still generating chunk #2.
  streamNextTime = audioCtx.currentTime + 0.14;
  ttsPlaybackStarted = true;
  setState('speaking');
  while(ttsPendingAudioChunks.length){
    queueAudioChunk(ttsPendingAudioChunks.shift());
  }
}
function pushBufferedTtsChunk(chunk){
  if(!chunk) return;
  if(ttsPlaybackStarted){
    queueAudioChunk(chunk);
    return;
  }
  ttsPendingAudioChunks.push(chunk);

  // Sentence-stream mode: server sends audio per completed sentence.
  // Start when the first sentence audio is ready, not after the whole answer.
  if(ttsPendingAudioChunks.length >= 1 || ttsAudioEnded){
    startBufferedTtsPlayback();
  } else if(!ttsBufferTimer){
    ttsBufferTimer = setTimeout(()=>startBufferedTtsPlayback(), 650);
  }
}


function trimAndFadePcm(f32, sampleRate){
  if(!f32 || !f32.length) return f32;
  const threshold = 0.006;
  const pad = Math.max(1, Math.floor(sampleRate * 0.018));
  let start = 0, end = f32.length - 1;

  while(start < f32.length && Math.abs(f32[start]) < threshold) start++;
  while(end > start && Math.abs(f32[end]) < threshold) end--;

  start = Math.max(0, start - pad);
  end = Math.min(f32.length - 1, end + pad);

  if(end <= start + sampleRate * 0.08) {
    // too short after trim; keep original to avoid cutting words
    start = 0; end = f32.length - 1;
  }

  let out = f32.subarray(start, end + 1);
  // Copy because subarray can keep large backing buffer and because we apply fades.
  out = new Float32Array(out);

  const fadeN = Math.min(Math.floor(sampleRate * 0.010), Math.floor(out.length / 4));
  for(let i=0;i<fadeN;i++){
    const g = i / Math.max(1, fadeN);
    out[i] *= g;
    out[out.length - 1 - i] *= g;
  }
  return out;
}

function stopPlayback({clearRequest=true}={}){
  for(const s of streamSources){
    try{ s.onended = null; }catch{}
    try{s.stop(0);}catch{}
    try{s.disconnect?.();}catch{}
  }
  streamSources=[];
  streamNextTime=0;
  resetTtsPlaybackBuffer();
  stopBrowserTts();
  if(clearRequest){ assistantOutputActive = false; if(assistantOutputResumeTimer){ clearTimeout(assistantOutputResumeTimer); assistantOutputResumeTimer=null; } if(assistantMuteWatchdogTimer){ clearTimeout(assistantMuteWatchdogTimer); assistantMuteWatchdogTimer=null; } assistantMuteToken += 1; }
  ttsEchoActive = false;
  ttsAudioEnded = false;
  ttsPlaybackStarted = false;
  currentAudioMessageId = null;
  if(clearRequest) currentAudioRequestId = null;
}
function startStreamPlayback(){ stopPlayback(); ensureAudioCtx(); speakingStartedAt = Date.now(); beginAssistantOutputMute('server_tts_playback'); streamNextTime=audioCtx.currentTime+.06; setState('speaking'); }
function queueAudioChunk(base64Pcm){
  ensureAudioCtx();
  const bin=atob(base64Pcm);
  const bytes=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
  const int16=new Int16Array(bytes.buffer);
  let f32=new Float32Array(int16.length);
  for(let i=0;i<int16.length;i++) f32[i]=int16[i]/32768;

  // Supertonic chunks often contain leading/trailing silence.
  // Trim it and add tiny fades so boundaries do not sound like abrupt stutters.
  f32 = trimAndFadePcm(f32, streamSampleRate);

  const buf=audioCtx.createBuffer(1,f32.length,streamSampleRate);
  buf.getChannelData(0).set(f32);
  const src=audioCtx.createBufferSource();
  src.buffer=buf;

  const gain=audioCtx.createGain();
  gain.gain.value=1.0;
  src.connect(gain);
  gain.connect(audioCtx.destination);
  if(analyser) gain.connect(analyser);

  // Keep a tiny overlap/lead instead of accidental dead gaps.
  const startAt=Math.max(streamNextTime - 0.006, audioCtx.currentTime + 0.010);
  src.start(startAt);
  streamNextTime=startAt+buf.duration;
  streamSources.push(src);
  src.onended=()=>{ const idx=streamSources.indexOf(src); if(idx>=0) streamSources.splice(idx,1); };
}

function setCameraEnabled(on){
  cameraEnabled = !!on;
  const btn = $('cameraToggle');
  if(btn){ btn.classList.toggle('active', cameraEnabled); btn.textContent = cameraEnabled ? 'Camera On' : 'Camera Off'; }
  const vid = $('cameraVideo');
  if(vid) vid.style.opacity = cameraEnabled ? 1 : .25;
  restartLiveFrameRecorder();
  updateLiveLabels();
}
$('cameraToggle').onclick = async()=>{ await startMediaIfNeeded(); setCameraEnabled(!cameraEnabled); };
$('screenShareBtn').onclick = startScreenShare; $('screenReplaceBtn').onclick = async()=>{ stopScreenShare(); await startScreenShare(); }; $('screenRemoveBtn').onclick = stopScreenShare; $('screenSendToggle').onclick=()=>{ screenSending=!screenSending; $('screenSendToggle').classList.toggle('active',screenSending); $('screenSendToggle').textContent=screenSending?'Screen On':'Screen Off'; updateLiveLabels(); };
async function startScreenShare(){ try{ screenStream=await navigator.mediaDevices.getDisplayMedia({video:{frameRate:{ideal:30,max:60}},audio:false}); $('screenVideo').srcObject=screenStream; $('screenVideo').style.display='block'; $('screenEmpty').style.display='none'; $('screenActions').style.display='flex'; $('screenBottom').style.display='flex'; screenSending=true; $('screenSendToggle').classList.add('active'); $('screenSendToggle').textContent='Screen On'; screenStream.getVideoTracks()[0]?.addEventListener('ended', stopScreenShare); restartLiveFrameRecorder(); if(appSettings.startAutoWithScreen) await setAutoMic(true); }catch(e){ console.warn(e); } }
function stopScreenShare(){ screenStream?.getTracks().forEach(t=>t.stop()); screenStream=null; liveFrameBuffers.screen=[]; $('screenVideo').srcObject=null; $('screenVideo').style.display='none'; $('screenEmpty').style.display='flex'; $('screenActions').style.display='none'; $('screenBottom').style.display='none'; updateLiveLabels(); }
function pickFile(accept, cb, multiple=false){ const input=document.createElement('input'); input.type='file'; input.accept=accept; input.multiple=!!multiple; input.onchange=()=>{ if(input.files?.length) cb(multiple ? input.files : input.files[0]); }; input.click(); }
$('pdfUploadBtn').onclick=()=>pickFile('application/pdf', loadPdf); $('pdfReplaceBtn').onclick=()=>pickFile('application/pdf', loadPdf); $('pdfRemoveBtn').onclick=removePdf; $('pdfPrevBtn').onclick=()=>changePdfPage(-1); $('pdfNextBtn').onclick=()=>changePdfPage(1); $('pdfSendToggle').onclick=()=>{ pdfSending=!pdfSending; $('pdfSendToggle').classList.toggle('active',pdfSending); $('pdfSendToggle').textContent=pdfSending?'PDF On':'PDF Off'; };
async function loadPdf(file){
  try{
    const pdfjs = await import('https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.mjs');
    pdfjs.GlobalWorkerOptions.workerSrc='https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.mjs';
    const bytes=await file.arrayBuffer(); pdfDoc=await pdfjs.getDocument({data:bytes}).promise; pdfPageCount=pdfDoc.numPages; pdfPage=1; await renderPdfPage();
    $('pdfCanvas').style.display='block'; $('pdfEmpty').style.display='none'; $('pdfActions').style.display='flex'; $('pdfBottom').style.display='flex'; pdfSending=true; $('pdfSendToggle').classList.add('active'); $('pdfSendToggle').textContent='PDF On';
  } catch(e){ alert('PDF failed to load: '+e.message); }
}
async function renderPdfPage(){ if(!pdfDoc) return; try{ pdfRenderTask?.cancel?.(); }catch{} const page=await pdfDoc.getPage(pdfPage); const viewport=page.getViewport({scale:1.45}); const c=$('pdfCanvas'); c.width=viewport.width; c.height=viewport.height; const ctx=c.getContext('2d'); pdfRenderTask=page.render({canvasContext:ctx,viewport}); await pdfRenderTask.promise.catch(()=>{}); $('pdfPageLabel').textContent=`${pdfPage}/${pdfPageCount}`; }
function changePdfPage(d){ if(!pdfDoc) return; pdfPage=Math.max(1,Math.min(pdfPageCount,pdfPage+d)); renderPdfPage(); }
function removePdf(){ try{pdfDoc?.destroy?.();}catch{} pdfDoc=null; pdfPage=1; pdfPageCount=0; $('pdfCanvas').style.display='none'; $('pdfEmpty').style.display='flex'; $('pdfActions').style.display='none'; $('pdfBottom').style.display='none'; }
$('videoUploadBtn').onclick=()=>pickFile('video/*', loadVideo); $('videoReplaceBtn').onclick=()=>pickFile('video/*', loadVideo); $('videoRemoveBtn').onclick=removeVideo; $('videoSendToggle').onclick=()=>{ videoSending=!videoSending; $('videoSendToggle').classList.toggle('active',videoSending); $('videoSendToggle').textContent=videoSending?'Video On':'Video Off'; };
function loadVideo(file){ if(videoObjectUrl) URL.revokeObjectURL(videoObjectUrl); videoObjectUrl=URL.createObjectURL(file); $('fileVideo').src=videoObjectUrl; $('fileVideo').style.display='block'; $('videoFileName').textContent=file.name; $('videoEmpty').style.display='none'; $('videoActions').style.display='flex'; $('videoBottom').style.display='flex'; videoSending=true; $('videoSendToggle').classList.add('active'); $('videoSendToggle').textContent='Video On'; liveFrameBuffers.video=[]; restartLiveFrameRecorder(); }
function removeVideo(){ if(videoObjectUrl) URL.revokeObjectURL(videoObjectUrl); videoObjectUrl=null; liveFrameBuffers.video=[]; $('fileVideo').removeAttribute('src'); $('fileVideo').load(); $('fileVideo').style.display='none'; $('videoEmpty').style.display='flex'; $('videoActions').style.display='none'; $('videoBottom').style.display='none'; updateLiveLabels(); }

$('imageUploadBtn')?.addEventListener('click',()=>pickFile('image/*', loadImages, true));
$('imageReplaceBtn')?.addEventListener('click',()=>pickFile('image/*', loadImages, true));
$('imageRemoveBtn')?.addEventListener('click',removeImages);
$('imageSendToggle')?.addEventListener('click',()=>{ imageSending=!imageSending; $('imageSendToggle').classList.toggle('active',imageSending); $('imageSendToggle').textContent=imageSending?'Images On':'Images Off'; updateLiveLabels(); });
async function loadImages(fileList){
  const files = Array.from(fileList instanceof FileList ? fileList : [fileList]).filter(f=>f && f.type && f.type.startsWith('image/'));
  if(!files.length) return;
  for(const u of imageObjectUrls) URL.revokeObjectURL(u);
  imageObjectUrls=[]; uploadedImages=[]; currentImageIndex=0;
  for(const f of files.slice(0,8)){
    const url=URL.createObjectURL(f); imageObjectUrls.push(url);
    const blob=await resizeImageBlobFromUrl(url, Number(appSettings?.imageMax||640), .82);
    if(blob) uploadedImages.push(blob);
  }
  if(imageObjectUrls[0]){ $('imagePreview').src=imageObjectUrls[0]; $('imagePreview').style.display='block'; }
  $('imageFileName').textContent = files.length === 1 ? files[0].name : `${files.length} images`;
  $('imageEmpty').style.display='none'; $('imageActions').style.display='flex'; $('imageBottom').style.display='flex';
  imageSending=true; $('imageSendToggle').classList.add('active'); $('imageSendToggle').textContent='Images On'; updateLiveLabels();
}
function removeImages(){ for(const u of imageObjectUrls) URL.revokeObjectURL(u); imageObjectUrls=[]; uploadedImages=[]; currentImageIndex=0; $('imagePreview').removeAttribute('src'); $('imagePreview').style.display='none'; $('imageEmpty').style.display='flex'; $('imageActions').style.display='none'; $('imageBottom').style.display='none'; updateLiveLabels(); }


let waveLastDrawAt = 0;
function drawWave(){ const nowDraw=performance.now(); if(nowDraw-waveLastDrawAt<80){ requestAnimationFrame(drawWave); return; } waveLastDrawAt=nowDraw; const c=$('waveform'), ctx=c.getContext('2d'), r=c.getBoundingClientRect(), dpr=window.devicePixelRatio||1; if(c.width!==Math.round(r.width*dpr)){ c.width=Math.round(r.width*dpr); c.height=Math.round(r.height*dpr); ctx.setTransform(dpr,0,0,dpr,0,0); } ctx.clearRect(0,0,r.width,r.height); const bars=16,gap=2,bw=(r.width-(bars-1)*gap)/bars,[color]=stateColors[state]||stateColors.loading; ctx.fillStyle=color; let data=null; if(analyser){ data=new Uint8Array(analyser.frequencyBinCount); analyser.getByteFrequencyData(data); } for(let i=0;i<bars;i++){ let amp=.05+Math.sin(Date.now()/450+i*.65)*.025; if(data) amp=Math.max(amp,data[Math.floor(i/bars*data.length*.6)]/255); const h=Math.max(2,amp*(r.height-4)); const x=i*(bw+gap), y=(r.height-h)/2; ctx.globalAlpha=.35+amp*.65; ctx.beginPath(); ctx.roundRect(x,y,bw,h,Math.min(3,bw/2,h/2)); ctx.fill(); } ctx.globalAlpha=1; requestAnimationFrame(drawWave); }
function initSplit(){ const main=$('main'), split=$('split'); if(!split) return; split.addEventListener('pointerdown',e=>{ splitDragging=true; split.setPointerCapture(e.pointerId); }); window.addEventListener('pointerup',()=>splitDragging=false); window.addEventListener('pointermove',e=>{ if(!splitDragging || window.innerWidth<981) return; const rect=main.getBoundingClientRect(); const ratio=(e.clientX-rect.left)/rect.width; const left=Math.max(.32, Math.min(.72, ratio)); main.style.gridTemplateColumns=`minmax(330px, ${left}fr) 10px minmax(420px, ${1-left}fr)`; }); }


// ── v17 overrides: chats save, message controls, replay/continue/regenerate, better live barge-in ──
function getMessageById(id){ return messages.find(x=>x.id===id); }
function historyFromList(list){
  return list.filter(m=>!m.pending && (m.role==='user'||m.role==='assistant') && textOfMessage(m)).map(m=>({role:m.role,text:textOfMessage(m)}));
}
function buildHistoryUntilIndex(endExclusive){
  return historyFromList(messages.slice(0, Math.max(0, endExclusive)));
}
function previousUserMessageIndex(idx){
  for(let i=idx-1;i>=0;i--) if(messages[i].role==='user' && textOfMessage(messages[i])) return i;
  return -1;
}
function replayAssistantMessage(id){
  const m=getMessageById(id);
  if(!m) return;
  const cached = messageAudioCache.get(id) || (Array.isArray(m.audioChunks) && m.audioChunks.length ? {chunks:m.audioChunks, sampleRate:m.audioSampleRate||24000} : null);
  if(cached && cached.chunks && cached.chunks.length){
    streamSampleRate = cached.sampleRate || 24000;
    startStreamPlayback();
    for(const chunk of cached.chunks) queueAudioChunk(chunk);
    return;
  }

  const text = textOfMessage(m);
  if(!text) return;
  if(ttsMode() === 'browser' && browserTtsSupported()){
    stopBrowserTts();
    browserTtsBuffer = text;
    enqueueBrowserTts('', {force:true});
    return;
  }
  if(ws && ws.readyState === WebSocket.OPEN){
    const requestId = `tts-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    replayTargets.set(requestId, id);
    ws.send(JSON.stringify({type:'tts', request_id:requestId, chat_id:activeChatId, text, voice: appSettings.voice || '', tts_engine: appSettings.ttsEngine || 'supertonic', silero_speaker: appSettings.sileroSpeaker || 'baya', silero_speed: Number(appSettings.sileroSpeed || 1.0)}));
    setStatus('processing','TTS replay');
  } else if('speechSynthesis' in window){
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }
}
function continueAssistantMessage(id){
  const idx = messages.findIndex(m=>m.id===id); if(idx<0) return;
  const hist = buildHistoryUntilIndex(idx+1);
  sendPayload({text:'Continue the previous answer from where it stopped. Do not repeat the beginning.', historyOverride: hist, suppressUserEcho:true});
}
function regenerateAssistantMessage(id){
  const idx = messages.findIndex(m=>m.id===id); if(idx<0) return;
  const userIdx = previousUserMessageIndex(idx);
  if(userIdx < 0) return;
  const promptText = textOfMessage(messages[userIdx]);
  const hist = buildHistoryUntilIndex(userIdx);
  sendPayload({text:promptText, historyOverride: hist, suppressUserEcho:true, replaceAssistantId:id});
}
function requestInterrupt(fromVoice=false){
  const rid = activeRequestId || currentAudioRequestId || null;

  // Hard client-side cancel: server may still send already generated chunks.
  // Mark BOTH text and audio request ids as ignored before stopping playback.
  if(activeRequestId){
    ignoredTextRequests.add(activeRequestId);
    ignoredAudioRequests.add(activeRequestId);
  }
  if(currentAudioRequestId){
    ignoredTextRequests.add(currentAudioRequestId);
    ignoredAudioRequests.add(currentAudioRequestId);
  }
  if(rid){
    ignoredTextRequests.add(rid);
    ignoredAudioRequests.add(rid);
  }

  try{ if(ws && ws.readyState===1) ws.send(JSON.stringify({type:'interrupt', chat_id:activeChatId, request_id: rid, reason: fromVoice ? 'barge_in' : 'manual'})); }catch{}

  stopPlayback({clearRequest:true});
  activeRequestId = null;
  currentAudioRequestId = null;
  currentAudioMessageId = null;
  activeAssistantId = null;
  ttsEchoActive = false;
  speechTranscript = '';
  clearAutoSttDraft();

  if(fromVoice){
    setState('listening');
    setStatus('connected','Interrupted · listening');
    if(autoMicEnabled){
      autoSttSending=false;
      autoSttSendingAt=0;
      restartBrowserSttSoon(120);
    }
  } else {
    setState('listening');
    setStatus('connected','Interrupted');
  }
}
renderMessages = function(){
  const box = $('messages'); box.innerHTML = '';
  for (const m of messages) {
    const div = document.createElement('div');
    div.className = `msg ${m.role}${m.pending?' pending':''}`;
    div.dataset.id = m.id; if(m.request_id) div.dataset.requestId = m.request_id;
    const body = m.pending ? '<span class="loading-dots"><span></span><span></span><span></span></span>' : escapeHtml(m.text);
    const meta = m.meta ? `<div class="meta">${escapeHtml(m.meta)}</div>` : '';
    let actions = `<button class="msg-action edit" title="edit">✎</button><button class="msg-action del" title="delete">×</button>`;
    if(m.role === 'assistant'){
      const canReplay = !!textOfMessage(m);
      actions = `<button class="msg-action replay" title="listen again" ${canReplay?'':'disabled'}>🔊</button><button class="msg-action cont" title="continue">▶</button><button class="msg-action regen" title="regenerate">↻</button>` + actions;
    }
    div.innerHTML = `<div class="msg-actions">${actions}</div><div class="msg-body">${body}</div>${meta}`;
    box.appendChild(div);
  }
  $('transcript').scrollTop = $('transcript').scrollHeight;
  saveActiveChat();
};
addMessage = function(role, text, meta='', pending=false, extra={}){ const m={id:nowId(),role,text,meta,pending,audioChunks:[],audioSampleRate:0,...(extra||{})}; messages.push(m); renderMessages(); return m.id; };
updateMessage = function(id, patch){ const m=messages.find(x=>x.id===id); if(m){ Object.assign(m,patch); renderMessages(); } };
deleteMessage = function(id){ messages = messages.filter(m => m.id !== id); renderMessages(); };
editMessage = function(id){ const m=messages.find(x=>x.id===id); if(!m) return; const next = prompt('Edit message:', m.text || ''); if(next!==null){ m.text = next; m.pending=false; renderMessages(); } };
historyForServer = function(){ const n=Math.max(2, Math.min(200, Number(appSettings.historyMessages||64))); return messages.filter(m=>!m.pending && (m.role==='user'||m.role==='assistant') && textOfMessage(m)).slice(-n).map(m=>({role:m.role,text:textOfMessage(m)})); };
$('messages').onclick = (e)=>{ const btn=e.target.closest('button'); if(!btn) return; const id=Number(e.target.closest('.msg')?.dataset.id); if(!id) return; if(btn.classList.contains('del')) deleteMessage(id); else if(btn.classList.contains('edit')) editMessage(id); else if(btn.classList.contains('replay')) replayAssistantMessage(id); else if(btn.classList.contains('cont')) continueAssistantMessage(id); else if(btn.classList.contains('regen')) regenerateAssistantMessage(id); };

startMedia = async function(){
  if(mediaStream && mediaStream.getTracks().length) return;
  const tracks = [];
  let gotVideo = false, gotAudio = false;
  try {
    const a = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
    a.getAudioTracks().forEach(t=>tracks.push(t)); gotAudio = true;
  } catch(e) { console.warn('audio failed', e?.message || e); }
  try {
    const v = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:424,max:640},height:{ideal:240,max:360},frameRate:{ideal:10,max:15},facingMode:'user'}});
    v.getVideoTracks().forEach(t=>tracks.push(t)); gotVideo = true;
  } catch(e) { console.warn('video failed', e?.message || e); }
  mediaStream = new MediaStream(tracks);
  if(gotVideo){ $('cameraVideo').srcObject = mediaStream; setCameraEnabled(true); }
  else { cameraEnabled = false; $('cameraToggle').textContent = 'No Camera'; $('cameraToggle').classList.remove('active'); }
  if(!gotAudio) console.warn('Microphone is unavailable or permission was denied');
};

sendPayload = function({text='', audio=null, transcript='', historyOverride=null, suppressUserEcho=false, replaceAssistantId=null}={}){
  if(!ws || ws.readyState!==WebSocket.OPEN){
    console.warn('WS not open; dropped request');
    setStatus('disconnected','Disconnected');
    return;
  }

  const hist = Array.isArray(historyOverride) ? historyOverride : historyForServer();
  const spokenText = (transcript || '').trim();
  const backendMode = appSettings.backend || 'llama_cpp';
  const micModeRaw = appSettings.audioInputMode || (Number(appSettings.sendRawAudio || 0) ? 'native' : 'stt');
  const micMode = backendMode === 'litertlm' ? 'native' : (backendMode === 'llama_native' ? 'native' : (['stt','native','hybrid'].includes(micModeRaw) ? (micModeRaw === 'hybrid' ? 'stt' : micModeRaw) : 'stt'));
  const hasAudio = !!audio;

  let outgoingText = '';
  if(text) outgoingText = String(text).trim();
  else outgoingText = spokenText;

  // Critical rule for llama.cpp/GGUF live mode:
  // if Browser STT produced a transcript, use that transcript as the chat text
  // and do NOT also attach raw WAV. Raw native audio is only a fallback when
  // STT failed/was disabled. This keeps the UI readable and avoids Gemma
  // answering helper/audio instructions instead of the user's question.
  const attachRawAudio = hasAudio && micMode === 'native' && (backendMode === 'litertlm' || backendMode === 'llama_native' || !outgoingText);

  if(!outgoingText && !attachRawAudio){
    console.warn('Dropped voice event: no STT text and raw audio is disabled', {micMode, hasAudio});
    setState('listening');
    setStatus('connected','Connected');
    return;
  }

  // v10 safety: do not let assistant TTS text become a new user turn.
  // This is intentionally checked here too, not only inside SpeechRecognition handlers,
  // because server ASR / delayed STT can reach sendPayload after the original mute window.
  if(outgoingText && autoMicEnabled && !suppressUserEcho && !replaceAssistantId && isLikelyAssistantEcho(outgoingText)){
    console.warn('[v10] Dropped likely assistant echo before sendPayload:', outgoingText);
    clearAutoSttDraft({toListening:true});
    setState('listening');
    setStatus('connected','Echo ignored · listening');
    if(autoMicEnabled) restartBrowserSttSoon(500);
    return;
  }

  if((activeRequestId || currentAudioRequestId || streamSources.length || ttsPendingAudioChunks.length || ttsPlaybackStarted || state === 'speaking') && !replaceAssistantId){
    requestInterrupt(false);
  }

  const wantsVision = visualIntent(outgoingText);
  const visionMode = appSettings.visionMode || 'auto';
  // Simple rule: Camera On means a camera frame can be attached automatically.
  // Screen/PDF/video are still attached only when the text asks for visual context.
  const includeCamera = !!cameraEnabled && visionMode !== 'never';
  const useVision = includeCamera || visionMode === 'always' || (visionMode === 'auto' && wantsVision);
  const frames = useVision ? collectFrameSequence({
    includeScreen: visionMode === 'always' || wantsVision,
    includeVideo: wantsVision,
    includePdf: wantsVision,
    includeCamera,
  }) : [];
  const images = useVision ? collectUploadedImages() : [];
  const used = [...new Set([...frames.map(x=>x.source), ...images.map(x=>x.source)])].join(' · ');
  activeRequestId = `r-${Date.now()}`;

  if(!suppressUserEcho){
    if(outgoingText) {
      const meta = text ? (used ? `with ${used}` : '') : (attachRawAudio ? `voice/${backendMode === 'litertlm' ? 'LiteRTLM' : micMode}${used ? ' · ' + used : ''}` : (used ? `voice/STT · ${used}` : 'voice/STT'));
      addMessage('user', outgoingText, meta, false, {request_id: activeRequestId});
      maybeTitleChat(outgoingText);
    } else if(attachRawAudio) {
      addMessage('user','[raw audio fallback sent]', used ? `voice/native fallback · ${used}` : `voice/native fallback`, false, {request_id: activeRequestId});
    } else if(frames.length || images.length) {
      addMessage('user','look at the attachments', used ? `with ${used}` : 'with image', false, {request_id: activeRequestId});
    }
  }

  stopBrowserTts();
  browserTtsBuffer = '';
  browserTtsQueue = [];
  if(replaceAssistantId){
    activeAssistantId = replaceAssistantId;
    updateMessage(replaceAssistantId,{text:'', meta:'', pending:true, audioChunks:[], audioSampleRate:0});
  } else {
    activeAssistantId = addMessage('assistant','', '', true);
  }

  const audioSeconds = attachRawAudio && audio ? estimateWavSecondsFromBase64(audio) : 0;
  console.debug('[TX USER TURN]', {
    requestId: activeRequestId,
    micMode,
    hasAudio,
    attachRawAudio,
    outgoingTextLen: outgoingText.length,
    transcriptLen: spokenText.length,
    audioB64Len: attachRawAudio && audio ? audio.length : 0,
    audioSeconds
  });

  setState('processing');
  setStatus('processing', attachRawAudio ? `Processing voice/${micMode}` : 'Processing');

  ws.send(JSON.stringify({
    type: outgoingText ? 'text' : (attachRawAudio ? 'audio' : 'image'),
    request_id: activeRequestId,
    chat_id: activeChatId,
    text: outgoingText,
    transcription: spokenText,
    audio: attachRawAudio ? audio : null,
    backend: backendMode,
    audio_mode: micMode,
    audio_seconds: audioSeconds,
    frames,
    images,
    lang:'en',
    voice: appSettings.voice || '',
    system_prompt:$('systemPrompt').value || '',
    history: hist,
    settings:{
      temperature:Number(appSettings.temperature),
      top_p:Number(appSettings.top_p),
      top_k:Number(appSettings.top_k),
      max_output_tokens:Number(appSettings.max_output_tokens),
      enable_thinking:false,
      tts_mode:ttsMode(),
      tts_engine:appSettings.ttsEngine||'supertonic',
      voice:appSettings.voice||'',
      silero_speaker:appSettings.sileroSpeaker||'baya',
      silero_speed:Number(appSettings.sileroSpeed||1.0),
      backend: backendMode,
      audio_input_mode: micMode,
      server_asr: Number(appSettings.serverAsr ?? 1)
    }
  }));
};
$('chatsBtn')?.addEventListener('click', openChats);
$('closeChatsBtn')?.addEventListener('click', closeChats);
$('drawerBackdrop')?.addEventListener('click', closeChats);
$('newChatBtn')?.addEventListener('click', async()=>{ const c=createChatRecord('New chat'); activeChatId=c.id; localStorage.setItem(ACTIVE_CHAT_KEY, activeChatId); loadActiveChat(); closeChats(); await startMediaIfNeeded(); setCameraEnabled(true); });
$('renameChatBtn')?.addEventListener('click', ()=>{ const c=currentChat(); const name=prompt('Chat name:', c.title||''); if(name!==null && name.trim()){ c.title=name.trim(); c.updatedAt=Date.now(); persistChats(); renderChatList(); } });
$('deleteChatBtn')?.addEventListener('click', ()=>{ if(chats.length<=1){ messages=[]; currentChat().messages=[]; renderMessages(); return; } const c=currentChat(); if(confirm(`Delete chat "${c.title||'New chat'}"?`)){ chats=chats.filter(x=>x.id!==activeChatId); activeChatId=chats[0].id; localStorage.setItem(ACTIVE_CHAT_KEY, activeChatId); persistChats(); loadActiveChat(); closeChats(); } });
$('settingsBtn')?.addEventListener('click', openSettings);
$('closeSettingsBtn')?.addEventListener('click', closeSettings);
$('settingsBackdrop')?.addEventListener('click', closeSettings);
$('cancelSettingsBtn')?.addEventListener('click', closeSettings);
$('saveSettingsBtn')?.addEventListener('click', ()=>{ const oldWs=getActiveWsUrl(); readSettingsForm(); closeSettings(); if(getActiveWsUrl()!==oldWs){ try{ ws?.close(); }catch{} connect(); } });
$('resetSettingsBtn')?.addEventListener('click', ()=>{ appSettings={...DEFAULT_SETTINGS}; saveSettings(); fillSettingsForm(); setCameraEnabled(true); });
function applyStablePreset(lang){
  appSettings = {...appSettings, backend:'llama_cpp', browserStt:1, serverAsr:0, sttLang:lang, audioInputMode:'stt', sendRawAudio:0, cameraMax:288, liveFps:0.2, liveFrames:1, visionMode:'auto', bargeIn:1};
  saveSettings();
  fillSettingsForm();
  setCameraEnabled(true);
  setStatus('connected', `Stable preset ${lang}`);
}
$('presetRuBtn')?.addEventListener('click', ()=>applyStablePreset('ru-RU'));
$('presetEnBtn')?.addEventListener('click', ()=>applyStablePreset('en-US'));
function applyLitertPreset(){
  appSettings = {...appSettings, backend:'litertlm', browserStt:0, serverAsr:0, sttLang:'ru-RU', audioInputMode:'native', sendRawAudio:1, cameraMax:320, liveFps:0.2, liveFrames:1, visionMode:'auto', bargeIn:1};
  saveSettings(); fillSettingsForm(); setCameraEnabled(true); setStatus('connected','LiteRTLM Parlor mode');
}
$('presetLitertBtn')?.addEventListener('click', applyLitertPreset);

$('systemPrompt')?.addEventListener('input', saveActiveChat);

const oldReset = $('resetBtn').onclick;
$('resetBtn').onclick = async () => { messages=[]; currentChat().messages=[]; renderMessages(); activeAssistantId=null; if(ws&&ws.readyState===1) ws.send(JSON.stringify({type:'reset', chat_id:activeChatId})); stopPlayback(); await startMediaIfNeeded(); setCameraEnabled(true); setState('listening'); };

(async function init(){ setState('loading'); loadActiveChat(); await startMedia(); setCameraEnabled(mediaStream?.getVideoTracks?.().length > 0); connect(); initSplit(); drawWave(); restartLiveFrameRecorder(); updateAutoMicUI(); if(appSettings.autoVoice) setTimeout(()=>setAutoMic(true), 700); setTimeout(()=>{ setState('listening'); if(autoMicEnabled) ensureAutoMicListening(); },500); })();
