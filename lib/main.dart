import 'package:flutter/material.dart';
import 'audio/audio_recorder.dart';

void main() {
  runApp(const ITantraApp());
}

class ITantraApp extends StatelessWidget {
  const ITantraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'iTantra',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
      ),
      home: const VoiceTestPage(),
    );
  }
}

class VoiceTestPage extends StatefulWidget {
  const VoiceTestPage({super.key});

  @override
  State<VoiceTestPage> createState() => _VoiceTestPageState();
}

class _VoiceTestPageState extends State<VoiceTestPage> {
  final AudioRecorderService _audioRecorder = AudioRecorderService();

  bool _isRecording = false;
  String _status = 'READY';
  String? _recordingPath;

  Future<void> _startRecording() async {
    try {
      final permission = await _audioRecorder.hasPermission();

      if (!permission) {
        setState(() {
          _status = 'Microphone permission denied';
        });
        return;
      }

      await _audioRecorder.startRecording();

      setState(() {
        _isRecording = true;
        _status = 'LISTENING';
        _recordingPath = null;
      });
    } catch (e) {
      setState(() {
        _status = 'ERROR: $e';
      });
    }
  }

  Future<void> _stopRecording() async {
    try {
      final path = await _audioRecorder.stopRecording();

      setState(() {
        _isRecording = false;
        _status = 'RECORDED';
        _recordingPath = path;
      });

      debugPrint('Recording saved to: $path');
    } catch (e) {
      setState(() {
        _isRecording = false;
        _status = 'ERROR: $e';
      });
    }
  }

  @override
  void dispose() {
    _audioRecorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('iTantra')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              _status,
              style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),

            const SizedBox(height: 40),

            GestureDetector(
              onLongPressStart: (_) {
                _startRecording();
              },
              onLongPressEnd: (_) {
                _stopRecording();
              },
              child: Container(
                width: 180,
                height: 180,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _isRecording ? Colors.red : Colors.indigo,
                ),
                child: const Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.mic, size: 60, color: Colors.white),
                    SizedBox(height: 12),
                    Text(
                      'HOLD TO TALK',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 40),

            if (_recordingPath != null)
              Padding(
                padding: const EdgeInsets.all(20),
                child: Text(
                  'Recording saved:\n$_recordingPath',
                  textAlign: TextAlign.center,
                ),
              ),
          ],
        ),
      ),
    );
  }
}
