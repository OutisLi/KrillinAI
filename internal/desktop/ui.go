package desktop

import (
	"fmt"
	"image/color"
	"krillin-ai/config"
	"krillin-ai/internal/deps"
	"krillin-ai/internal/server"
	"krillin-ai/internal/types"
	"krillin-ai/log"
	"path/filepath"
	"strconv"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/data/binding"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
	"go.uber.org/zap"
)

// 创建配置界面
func CreateConfigTab(window fyne.Window) fyne.CanvasObject {
	pageTitle := TitleText("应用配置")

	appGroup := createAppConfigGroup()
	serverGroup := createServerConfigGroup()
	llmGroup := createLlmConfigGroup()
	transcribeGroup := createTranscribeConfigGroup()
	ttsGroup := createTtsConfigGroup()

	var background *canvas.LinearGradient
	if GetCurrentThemeIsDark() {
		background = canvas.NewLinearGradient(
			color.NRGBA{R: 15, G: 23, B: 42, A: 255},
			color.NRGBA{R: 30, G: 41, B: 59, A: 255},
			0.0,
		)
	} else {
		background = canvas.NewLinearGradient(
			color.NRGBA{R: 248, G: 250, B: 252, A: 255},
			color.NRGBA{R: 241, G: 245, B: 249, A: 255},
			0.0,
		)
	}

	spacer1 := canvas.NewRectangle(color.NRGBA{R: 0, G: 0, B: 0, A: 0})
	spacer1.SetMinSize(fyne.NewSize(0, 15))
	spacer2 := canvas.NewRectangle(color.NRGBA{R: 0, G: 0, B: 0, A: 0})
	spacer2.SetMinSize(fyne.NewSize(0, 15))

	configContainer := container.NewVBox(
		container.NewPadded(pageTitle),
		spacer1,
		container.NewPadded(appGroup),
		container.NewPadded(serverGroup),
		container.NewPadded(llmGroup),
		container.NewPadded(transcribeGroup),
		container.NewPadded(ttsGroup),
		spacer2,
	)

	scroll := container.NewScroll(configContainer)

	configStack := container.NewStack(background, scroll)

	return container.NewPadded(configStack)
}

// 创建字幕任务界面
func CreateSubtitleTab(window fyne.Window) fyne.CanvasObject {
	sm := NewSubtitleManager(window)

	title1 := TitleText("视频翻译配音")
	title2 := TitleText("Video Translate & Dubbing")
	titleContainer := container.NewVBox(title1, title2)

	videoInputContainer := createVideoInputContainer(sm)
	subtitleSettingsCard := createSubtitleSettingsCard(sm)
	voiceSettingsCard := createVoiceSettingsCard(sm)
	embedSettingsCard := createEmbedSettingsCard(sm)

	progress, downloadContainer, tipsLabel := createProgressAndDownloadArea(sm)

	startButton := createStartButton(window, sm, videoInputContainer, embedSettingsCard, progress, downloadContainer)
	startButtonContainer := container.NewHBox(layout.NewSpacer(), startButton, layout.NewSpacer())

	var background *canvas.LinearGradient
	if GetCurrentThemeIsDark() {
		background = canvas.NewLinearGradient(
			color.NRGBA{R: 15, G: 23, B: 42, A: 255},
			color.NRGBA{R: 30, G: 41, B: 59, A: 255},
			0.0,
		)
	} else {
		background = canvas.NewLinearGradient(
			color.NRGBA{R: 248, G: 250, B: 252, A: 255},
			color.NRGBA{R: 241, G: 245, B: 249, A: 255},
			0.0,
		)
	}

	spacer1 := canvas.NewRectangle(color.NRGBA{R: 0, G: 0, B: 0, A: 0})
	spacer1.SetMinSize(fyne.NewSize(0, 15))
	spacer2 := canvas.NewRectangle(color.NRGBA{R: 0, G: 0, B: 0, A: 0})
	spacer2.SetMinSize(fyne.NewSize(0, 15))
	spacer3 := canvas.NewRectangle(color.NRGBA{R: 0, G: 0, B: 0, A: 0})
	spacer3.SetMinSize(fyne.NewSize(0, 15))

	progressArea := container.NewVBox(progress)

	mainContent := container.NewVBox(
		container.NewPadded(titleContainer),
		spacer1,
		container.NewPadded(videoInputContainer),
		container.NewPadded(subtitleSettingsCard),
		container.NewPadded(voiceSettingsCard),
		container.NewPadded(embedSettingsCard),
		spacer2,
		container.NewPadded(startButtonContainer),
		spacer3,
		progressArea,
		downloadContainer,
		tipsLabel,
	)

	scroll := container.NewScroll(mainContent)

	// 使用一个Stack将背景和滚动内容组合
	contentStack := container.NewStack(background, scroll)

	return container.NewPadded(contentStack)
}

// 创建应用配置组
func createAppConfigGroup() *fyne.Container {
	appSegmentDurationEntry := StyledEntry("字幕分段处理时长(分钟)")
	appSegmentDurationEntry.Bind(binding.IntToString(binding.BindInt(&config.Conf.App.SegmentDuration)))
	appSegmentDurationEntry.Validator = func(s string) error {
		val, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("请输入数字")
		}
		if val < 1 || val > 30 {
			return fmt.Errorf("请输入1-30之间的数字")
		}
		return nil
	}

	appTranscribeParallelNumEntry := StyledEntry("转录并行数量")
	appTranscribeParallelNumEntry.Bind(binding.IntToString(binding.BindInt(&config.Conf.App.TranscribeParallelNum)))
	appTranscribeParallelNumEntry.Validator = func(s string) error {
		val, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("请输入数字")
		}
		if val < 1 || val > 10 {
			return fmt.Errorf("请输入1-10之间的数字")
		}
		return nil
	}

	appTranslateParallelNumEntry := StyledEntry("翻译并行数量")
	appTranslateParallelNumEntry.Bind(binding.IntToString(binding.BindInt(&config.Conf.App.TranslateParallelNum)))
	appTranslateParallelNumEntry.Validator = func(s string) error {
		val, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("请输入数字")
		}
		if val < 1 || val > 20 {
			return fmt.Errorf("请输入1-20之间的数字")
		}
		return nil
	}

	appTranscribeMaxAttemptsEntry := StyledEntry("转录最大尝试次数")
	appTranscribeMaxAttemptsEntry.Bind(binding.IntToString(binding.BindInt(&config.Conf.App.TranscribeMaxAttempts)))
	appTranscribeMaxAttemptsEntry.Validator = func(s string) error {
		val, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("请输入数字")
		}
		if val < 1 || val > 10 {
			return fmt.Errorf("请输入1-10之间的数字")
		}
		return nil
	}

	appTranslateMaxAttemptsEntry := StyledEntry("翻译最大尝试次数")
	appTranslateMaxAttemptsEntry.Bind(binding.IntToString(binding.BindInt(&config.Conf.App.TranslateMaxAttempts)))
	appTranslateMaxAttemptsEntry.Validator = func(s string) error {
		val, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("请输入数字")
		}
		if val < 1 || val > 20 {
			return fmt.Errorf("请输入1-20之间的数字")
		}
		return nil
	}

	appMaxSentenceLengthEntry := StyledEntry("每个句子最大字符数 Max sentence length")
	appMaxSentenceLengthEntry.Bind(binding.IntToString(binding.BindInt(&config.Conf.App.MaxSentenceLength)))
	appMaxSentenceLengthEntry.Validator = func(s string) error {
		val, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("请输入数字")
		}
		if val < 1 || val > 200 {
			return fmt.Errorf("请输入1-200之间的数字")
		}
		return nil
	}

	appProxyEntry := StyledEntry("网络代理地址")
	appProxyEntry.Bind(binding.BindString(&config.Conf.App.Proxy))

	form := widget.NewForm(
		widget.NewFormItem("分段处理时长(分钟) Segment duration (minutes)", appSegmentDurationEntry),
		widget.NewFormItem("转录最大并行数量 Transcribe parallel num", appTranscribeParallelNumEntry),
		widget.NewFormItem("翻译最大并行数量 Translate parallel num", appTranslateParallelNumEntry),
		widget.NewFormItem("转录最大尝试次数 Transcribe max attempts", appTranscribeMaxAttemptsEntry),
		widget.NewFormItem("翻译最大尝试次数 Translate max attempts", appTranslateMaxAttemptsEntry),
		widget.NewFormItem("每个句子最大字符数 Max sentence length", appMaxSentenceLengthEntry),
		widget.NewFormItem("网络代理地址 proxy", appProxyEntry),
	)

	return GlassmorphismCard("应用配置 App Config", "基本参数 Basic config", form, GetCurrentThemeIsDark())
}

// 创建server配置组
func createServerConfigGroup() *fyne.Container {
	serverHostEntry := StyledEntry("服务器地址 Server address")
	serverHostEntry.Bind(binding.BindString(&config.Conf.Server.Host))

	serverPortEntry := StyledEntry("服务器端口 Server port")
	serverPortEntry.Bind(binding.IntToString(binding.BindInt(&config.Conf.Server.Port)))
	serverPortEntry.Validator = func(s string) error {
		val, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("请输入数字")
		}
		if val < 1 || val > 65535 {
			return fmt.Errorf("请输入1-65535之间的有效端口")
		}
		return nil
	}

	form := widget.NewForm(
		widget.NewFormItem("服务器地址 Server address", serverHostEntry),
		widget.NewFormItem("服务器端口 Server port", serverPortEntry),
	)

	return GlassmorphismCard("服务器配置 Server Config", "API服务器设置 API server settings", form, GetCurrentThemeIsDark())
}

// 创建LLM配置组
func createLlmConfigGroup() *fyne.Container {
	baseUrlEntry := StyledEntry("API Base URL")
	baseUrlEntry.Bind(binding.BindString(&config.Conf.Llm.BaseUrl))

	apiKeyEntry := StyledPasswordEntry("API Key")
	apiKeyEntry.Bind(binding.BindString(&config.Conf.Llm.ApiKey))

	modelEntry := StyledEntry("模型名称 Model name")
	modelEntry.Bind(binding.BindString(&config.Conf.Llm.Model))

	form := widget.NewForm(
		widget.NewFormItem("API Base URL", baseUrlEntry),
		widget.NewFormItem("API Key", apiKeyEntry),
		widget.NewFormItem("模型名称 Model name", modelEntry),
	)
	return GlassmorphismCard("LLM 配置 LLM Config", "LLM配置 LLM config", form, GetCurrentThemeIsDark())
}

// 创建语音识别配置组
func createTranscribeConfigGroup() *fyne.Container {
	providerOptions := []string{"openai", "fasterwhisper", "whisperkit", "whispercpp", "aliyun"}
	providerSelect := widget.NewSelect(providerOptions, func(value string) {
		config.Conf.Transcribe.Provider = value
	})
	providerSelect.SetSelected(config.Conf.Transcribe.Provider)

	openaiBaseUrlEntry := StyledEntry("API Base URL")
	openaiBaseUrlEntry.Bind(binding.BindString(&config.Conf.Transcribe.Openai.BaseUrl))
	openaiApiKeyEntry := StyledPasswordEntry("API Key")
	openaiApiKeyEntry.Bind(binding.BindString(&config.Conf.Transcribe.Openai.ApiKey))
	openaiModelEntry := StyledEntry("模型名称 Model name")
	openaiModelEntry.Bind(binding.BindString(&config.Conf.Transcribe.Openai.Model))

	fasterWhisperModelEntry := StyledEntry("模型名称 Model name")
	fasterWhisperModelEntry.Bind(binding.BindString(&config.Conf.Transcribe.Fasterwhisper.Model))

	whisperKitModelEntry := StyledEntry("模型名称 Model name")
	whisperKitModelEntry.Bind(binding.BindString(&config.Conf.Transcribe.Whisperkit.Model))

	whisperCppModelEntry := StyledEntry("模型名称 Model name")
	whisperCppModelEntry.Bind(binding.BindString(&config.Conf.Transcribe.Whispercpp.Model))

	aliyunOssKeyIdEntry := StyledEntry("阿里云 Aliyun Access Key ID")
	aliyunOssKeyIdEntry.Bind(binding.BindString(&config.Conf.Transcribe.Aliyun.Oss.AccessKeyId))
	aliyunOssKeySecretEntry := StyledPasswordEntry("阿里云 Aliyun Access Key Secret")
	aliyunOssKeySecretEntry.Bind(binding.BindString(&config.Conf.Transcribe.Aliyun.Oss.AccessKeySecret))
	aliyunOssBucketEntry := StyledEntry("阿里云 Aliyun OSS Bucket名称")
	aliyunOssBucketEntry.Bind(binding.BindString(&config.Conf.Transcribe.Aliyun.Oss.Bucket))

	aliyunSpeechKeyIdEntry := StyledEntry("阿里云 Aliyun Speech Access Key ID")
	aliyunSpeechKeyIdEntry.Bind(binding.BindString(&config.Conf.Transcribe.Aliyun.Speech.AccessKeyId))
	aliyunSpeechKeySecretEntry := StyledPasswordEntry("阿里云 Aliyun Speech Access Key Secret")
	aliyunSpeechKeySecretEntry.Bind(binding.BindString(&config.Conf.Transcribe.Aliyun.Speech.AccessKeySecret))
	aliyunSpeechAppKeyEntry := StyledEntry("阿里云 Aliyun Speech App Key")
	aliyunSpeechAppKeyEntry.Bind(binding.BindString(&config.Conf.Transcribe.Aliyun.Speech.AppKey))

	form := widget.NewForm(
		widget.NewFormItem("提供商 Provider", providerSelect),
		widget.NewFormItem("GPU加速 GPU acceleration", widget.NewCheckWithData("启用 Enable", binding.BindBool(&config.Conf.Transcribe.EnableGpuAcceleration))),

		widget.NewFormItem("OpenAI Base URL", openaiBaseUrlEntry),
		widget.NewFormItem("OpenAI API Key", openaiApiKeyEntry),
		widget.NewFormItem("OpenAI 模型 Model", openaiModelEntry),

		widget.NewFormItem("FasterWhisper 模型 Model", fasterWhisperModelEntry),

		widget.NewFormItem("WhisperKit 模型 Model", whisperKitModelEntry),

		widget.NewFormItem("WhisperCpp 模型 Model", whisperCppModelEntry),

		widget.NewFormItem("阿里云 Aliyun OSS Access Key ID", aliyunOssKeyIdEntry),
		widget.NewFormItem("阿里云 Aliyun OSS Access Key Secret", aliyunOssKeySecretEntry),
		widget.NewFormItem("阿里云 Aliyun OSS Bucket Name", aliyunOssBucketEntry),

		widget.NewFormItem("阿里云语音 Aliyun Speech Access Key ID", aliyunSpeechKeyIdEntry),
		widget.NewFormItem("阿里云语音 Aliyun Speech Access Key Secret", aliyunSpeechKeySecretEntry),
		widget.NewFormItem("阿里云语音 Aliyun Speech App Key", aliyunSpeechAppKeyEntry),
	)

	return GlassmorphismCard("语音识别配置 Transcribe Config", "语音识别配置 Transcribe config", form, GetCurrentThemeIsDark())
}

// 创建文本转语音配置组
func createTtsConfigGroup() *fyne.Container {
	providerOptions := []string{"openai", "aliyun", "edge-tts"}
	providerSelect := widget.NewSelect(providerOptions, func(value string) {
		config.Conf.Tts.Provider = value
	})
	providerSelect.SetSelected(config.Conf.Tts.Provider)

	openaiBaseUrlEntry := StyledEntry("API Base URL")
	openaiBaseUrlEntry.Bind(binding.BindString(&config.Conf.Tts.Openai.BaseUrl))
	openaiApiKeyEntry := StyledPasswordEntry("API Key")
	openaiApiKeyEntry.Bind(binding.BindString(&config.Conf.Tts.Openai.ApiKey))
	openaiModelEntry := StyledEntry("模型名称 Model name")
	openaiModelEntry.Bind(binding.BindString(&config.Conf.Tts.Openai.Model))

	aliyunOssKeyIdEntry := StyledEntry("阿里云 Aliyun Access Key ID")
	aliyunOssKeyIdEntry.Bind(binding.BindString(&config.Conf.Tts.Aliyun.Oss.AccessKeyId))
	aliyunOssKeySecretEntry := StyledPasswordEntry("阿里云 Aliyun Access Key Secret")
	aliyunOssKeySecretEntry.Bind(binding.BindString(&config.Conf.Tts.Aliyun.Oss.AccessKeySecret))
	aliyunOssBucketEntry := StyledEntry("阿里云 Aliyun OSS Bucket名称")
	aliyunOssBucketEntry.Bind(binding.BindString(&config.Conf.Tts.Aliyun.Oss.Bucket))

	aliyunSpeechKeyIdEntry := StyledEntry("阿里云 Aliyun Speech Access Key ID")
	aliyunSpeechKeyIdEntry.Bind(binding.BindString(&config.Conf.Tts.Aliyun.Speech.AccessKeyId))
	aliyunSpeechKeySecretEntry := StyledPasswordEntry("阿里云 Aliyun Speech Access Key Secret")
	aliyunSpeechKeySecretEntry.Bind(binding.BindString(&config.Conf.Tts.Aliyun.Speech.AccessKeySecret))
	aliyunSpeechAppKeyEntry := StyledEntry("阿里云 Aliyun Speech App Key")
	aliyunSpeechAppKeyEntry.Bind(binding.BindString(&config.Conf.Tts.Aliyun.Speech.AppKey))

	form := widget.NewForm(
		widget.NewFormItem("提供商 Provider", providerSelect),

		widget.NewFormItem("OpenAI Base URL", openaiBaseUrlEntry),
		widget.NewFormItem("OpenAI API Key", openaiApiKeyEntry),
		widget.NewFormItem("OpenAI 模型 Model", openaiModelEntry),

		widget.NewFormItem("阿里云 Aliyun OSS Access Key ID", aliyunOssKeyIdEntry),
		widget.NewFormItem("阿里云 Aliyun OSS Access Key Secret", aliyunOssKeySecretEntry),
		widget.NewFormItem("阿里云 Aliyun OSS Bucket", aliyunOssBucketEntry),

		widget.NewFormItem("阿里云 Aliyun Speech Access Key ID", aliyunSpeechKeyIdEntry),
		widget.NewFormItem("阿里云 Aliyun  Speech Access Key Secret", aliyunSpeechKeySecretEntry),
		widget.NewFormItem("阿里云 Aliyun Speech App Key", aliyunSpeechAppKeyEntry),
	)

	return GlassmorphismCard("文本转语音配置 TTS Config", "文本转语音配置 TTS config", form, GetCurrentThemeIsDark())
}

// 创建视频输入容器
func createVideoInputContainer(sm *SubtitleManager) *fyne.Container {
	inputTypeRadio := widget.NewRadioGroup([]string{"本地上传 Upload a file", "输入链接 Paste a link"}, nil)
	inputTypeRadio.Horizontal = true
	inputTypeContainer := container.NewHBox(
		inputTypeRadio,
	)

	urlEntry := StyledEntry("输入视频链接 Paste a link here")
	urlEntry.Hide()
	urlEntry.OnChanged = func(text string) {
		sm.SetVideoUrl(text)
	}

	selectButton := PrimaryButton("选择视频文件 Select Video Files", theme.FolderOpenIcon(), sm.ShowFileDialog)

	selectedVideoLabel := widget.NewLabel("")
	selectedVideoLabel.Hide()

	sm.SetVideoSelectedCallback(func(path string) { // 设置视频地址+控制信息展示
		if path != "" {
			sm.SetVideoUrl(path)
			selectedVideoLabel.SetText("已选择Chosen: " + filepath.Base(path))
			selectedVideoLabel.Show()
		} else {
			selectedVideoLabel.Hide()
		}
	})

	sm.SetVideosSelectedCallback(func(paths []string) {
		if len(paths) > 0 {
			sm.SetVideoUrl(paths[0])

			fileNames := make([]string, 0, len(paths))
			for _, path := range paths {
				fileNames = append(fileNames, filepath.Base(path))
			}

			displayText := fmt.Sprintf("已选择 %d 个文件:\n", len(paths))
			for i, name := range fileNames {
				displayText += fmt.Sprintf("%d. %s\n", i+1, name)
			}

			selectedVideoLabel.SetText(displayText)
			selectedVideoLabel.Show()
		} else {
			selectedVideoLabel.Hide()
		}
	})

	videoInputContainer := container.NewVBox()
	videoInputContainer.Objects = []fyne.CanvasObject{selectButton, selectedVideoLabel}

	inputTypeRadio.SetSelected("本地上传 Upload a file")
	inputTypeRadio.OnChanged = func(value string) {
		if value == "本地上传 Upload a file" {
			urlEntry.Hide()
			selectButton.Show()
			selectedVideoLabel.Show()
			videoInputContainer.Objects = []fyne.CanvasObject{selectButton, selectedVideoLabel}
			sm.SetVideoUrl("")
		} else {
			selectButton.Hide()
			selectedVideoLabel.Hide()
			urlEntry.Show()
			videoInputContainer.Objects = []fyne.CanvasObject{urlEntry}
		}
		videoInputContainer.Refresh()
	}

	content := container.NewVBox(
		container.NewPadded(inputTypeContainer),
		container.NewPadded(videoInputContainer),
	)

	return GlassmorphismCard("1. 选择视频 Select Video", "", content, GetCurrentThemeIsDark())
}

// 创建字幕设置卡片
func createSubtitleSettingsCard(sm *SubtitleManager) *fyne.Container {
	positionSelect := widget.NewSelect([]string{
		"翻译在上 Translation Above",
		"翻译在下 Translation Below",
	}, func(value string) {
		if value == "翻译在上 Translation Above" {
			sm.SetBilingualPosition(1)
		} else {
			sm.SetBilingualPosition(2)
		}
	})
	positionSelect.SetSelected("翻译在上 Translation Above")

	bilingualCheck := widget.NewCheck("启用双语字幕 Bilingual Subtitles", func(checked bool) {
		sm.SetBilingualEnabled(checked)
		if checked {
			positionSelect.Enable()
		} else {
			positionSelect.Disable()
		}
	})
	bilingualCheck.SetChecked(true)

	var targetSelectOptions []string
	targetLangMap := make(map[string]string)
	for code, name := range types.StandardLanguageCode2Name {
		targetSelectOptions = append(targetSelectOptions, name)
		targetLangMap[name] = string(code)
	}
	targetLangSelector := StyledSelect(targetSelectOptions, func(value string) {
		sm.SetTargetLang(targetLangMap[value])
	})

	langContainer := container.NewVBox(
		container.NewHBox(
			widget.NewLabel("源语言 Original Language:"),
			StyledSelect([]string{
				"简体中文", "English", "日本語", "Türkçe", "Deutsch", "한국어", "Русский язык", "Bahasa Melayu",
			}, func(value string) {
				sourceLangMap := map[string]string{
					"简体中文": "zh_cn", "English": "en", "日本語": "ja",
					"Türkçe": "tr", "Deutsch": "de", "한국어": "ko", "Русский язык": "ru",
					"Bahasa Melayu": "ms",
				}
				sm.SetSourceLang(sourceLangMap[value])
			}),
		),
		container.NewHBox(
			widget.NewLabel("翻译成 Translate To:"),
			targetLangSelector,
		),
	)

	// 设置默认语言
	langContainer.Objects[0].(*fyne.Container).Objects[1].(*widget.Select).SetSelected("English")
	langContainer.Objects[1].(*fyne.Container).Objects[1].(*widget.Select).SetSelected("简体中文")

	fillerCheck := widget.NewCheck("启用语气词过滤 Tone Word Filtering", func(checked bool) {
		sm.SetFillerFilter(checked)
	})
	fillerCheck.SetChecked(true)

	content := container.NewVBox(
		container.NewHBox(bilingualCheck, fillerCheck),
		langContainer,
		positionSelect,
	)

	return ModernCard("2. 字幕设置 Subtitle settings", content, GetCurrentThemeIsDark())
}

// 创建配音设置卡片
func createVoiceSettingsCard(sm *SubtitleManager) *fyne.Container {
	voiceCodeEntry := widget.NewEntry()
	voiceCodeEntry.SetPlaceHolder("输入声音代码 Enter voice code")
	voiceCodeEntry.OnChanged = func(text string) {
		sm.SetTtsVoiceCode(text)
	}
	voiceCodeEntry.Disable()

	// 音色克隆功能 - 当前支持阿里云TTS，未来可扩展其他提供商
	audioSampleButton := SecondaryButton("选择音色克隆样本 Select Voice Clone Sample（Aliyun TTS Supported）", theme.MediaMusicIcon(), sm.ShowAudioFileDialog)
	audioSampleButton.Disable()

	voiceoverCheck := widget.NewCheck("启用配音 Apply Dubbing", func(checked bool) {
		sm.SetVoiceoverEnabled(checked)
		if checked {
			voiceCodeEntry.Enable()
			audioSampleButton.Enable()
		} else {
			voiceCodeEntry.Disable()
			audioSampleButton.Disable()
		}
	})

	grid := container.NewVBox(
		container.NewHBox(voiceoverCheck),
		container.NewHBox(container.NewBorder(voiceCodeEntry, nil, nil, audioSampleButton)),
	)

	return ModernCard("3. 配音设置 Dubbing settings", grid, GetCurrentThemeIsDark())
}

// 视频合成卡片
func createEmbedSettingsCard(sm *SubtitleManager) *fyne.Container {
	embedCheck := widget.NewCheck("合成 Composite", nil)

	embedTypeSelect := StyledSelect([]string{
		"横屏输出 Landscape（16：9）", "竖屏输出 Portrait（9:16）", "横屏+竖屏 (Landscape+Portrait)",
	}, nil)
	embedTypeSelect.Disable()

	mainTitleEntry := StyledEntry("请输入主标题 Enter main title")
	subTitleEntry := StyledEntry("请输入副标题 Enter sub title")

	titleInputContainer := container.NewVBox(
		container.NewGridWithColumns(2,
			widget.NewLabel("主标题 Main title:"),
			mainTitleEntry,
		),
		container.NewGridWithColumns(2,
			widget.NewLabel("副标题 Sub title:"),
			subTitleEntry,
		),
	)
	titleInputContainer.Hide()

	embedCheck.OnChanged = func(checked bool) {
		if checked {
			embedTypeSelect.Enable()
			embedTypeSelect.SetSelected("横屏输出 Landscape（16：9）")
		} else {
			embedTypeSelect.Disable()
			sm.SetEmbedSubtitle("none")
		}
	}

	embedTypeSelect.OnChanged = func(value string) {
		switch value {
		case "横屏输出 Landscape（16：9）":
			titleInputContainer.Hide()
			sm.SetEmbedSubtitle("horizontal")
		case "竖屏输出 Portrait（9:16）":
			titleInputContainer.Show()
			sm.SetEmbedSubtitle("vertical")
		case "横屏+竖屏 (Landscape+Portrait)":
			titleInputContainer.Show()
			sm.SetEmbedSubtitle("all")
		}
	}

	topContainer := container.NewHBox(embedCheck, embedTypeSelect)

	mainContainer := container.NewVBox(
		topContainer,
		container.NewPadded(titleInputContainer),
	)

	return ModernCard("视频合成设置 Composition Settings", mainContainer, GetCurrentThemeIsDark())
}

// 创建进度和下载区域
func createProgressAndDownloadArea(sm *SubtitleManager) (*widget.ProgressBar, *fyne.Container, *fyne.Container) {
	progress := widget.NewProgressBar()
	progress.Hide()

	percentLabel := widget.NewLabel("0%")
	percentLabel.Hide()
	percentLabel.Alignment = fyne.TextAlignTrailing

	progressContainer := container.NewBorder(nil, nil, nil, percentLabel, progress)
	progressContainer.Hide()

	progressBg := canvas.NewRectangle(color.NRGBA{R: 240, G: 245, B: 250, A: 230})
	progressBg.SetMinSize(fyne.NewSize(0, 40))
	progressBg.CornerRadius = 8

	progressShadow := canvas.NewRectangle(color.NRGBA{R: 0, G: 0, B: 0, A: 20})
	progressShadow.Move(fyne.NewPos(2, 2))
	progressShadow.SetMinSize(fyne.NewSize(0, 40))
	progressShadow.CornerRadius = 8

	progressWithBg := container.NewStack(
		progressShadow,
		progressBg,
		container.NewPadded(progressContainer),
	)
	progressWithBg.Hide()

	sm.SetProgressBar(progress)
	sm.SetProgressLabel(percentLabel)

	downloadBg := canvas.NewRectangle(color.NRGBA{R: 240, G: 250, B: 255, A: 230})
	downloadBg.CornerRadius = 10

	downloadContainer := container.NewVBox()
	downloadContainer.Hide()
	sm.SetDownloadContainer(downloadContainer)

	downloadWithBg := container.NewStack(
		downloadBg,
		container.NewPadded(downloadContainer),
	)
	downloadWithBg.Hide()

	tipsLabel := widget.NewLabel("")
	tipsLabel.Hide()
	tipsLabel.Alignment = fyne.TextAlignCenter
	tipsLabel.Wrapping = fyne.TextWrapWord
	sm.SetTipsLabel(tipsLabel)

	tipsBg := canvas.NewRectangle(color.NRGBA{R: 255, G: 250, B: 230, A: 200})
	tipsBg.CornerRadius = 6

	tipsWithBg := container.NewStack(
		tipsBg,
		container.NewPadded(tipsLabel),
	)
	tipsWithBg.Hide()

	return progress, downloadWithBg, tipsWithBg
}

// 开始按钮
func createStartButton(window fyne.Window, sm *SubtitleManager, videoInputContainer *fyne.Container, embedSettingsCard *fyne.Container, progress *widget.ProgressBar, downloadContainer *fyne.Container) *widget.Button {
	btn := widget.NewButtonWithIcon("开始翻译 Start Translating", theme.MediaPlayIcon(), nil)
	btn.Importance = widget.HighImportance

	btn.OnTapped = func() {
		originalImportance := btn.Importance
		btn.Importance = widget.DangerImportance
		btn.Refresh()

		go func() {
			time.Sleep(300 * time.Millisecond)
			btn.Importance = originalImportance
			btn.Refresh()
		}()

		var mainTitle, subTitle string

		if embedSettingsCard != nil && len(embedSettingsCard.Objects) > 1 {
			if titleContainer, ok := embedSettingsCard.Objects[1].(*fyne.Container); ok && titleContainer != nil && len(titleContainer.Objects) >= 2 {
				if mainTitleRow, ok := titleContainer.Objects[0].(*fyne.Container); ok && mainTitleRow != nil && len(mainTitleRow.Objects) >= 2 {
					if mainTitleEntry, ok := mainTitleRow.Objects[1].(*widget.Entry); ok {
						mainTitle = mainTitleEntry.Text
					}
				}

				if subTitleRow, ok := titleContainer.Objects[1].(*fyne.Container); ok && subTitleRow != nil && len(subTitleRow.Objects) >= 2 {
					if subTitleEntry, ok := subTitleRow.Objects[1].(*widget.Entry); ok {
						subTitle = subTitleEntry.Text
					}
				}
			}
		}

		sm.SetVerticalTitles(mainTitle, subTitle)

		progress.Show()
		sm.progressBar.SetValue(0)
		downloadContainer.Hide()

		if sm.GetVideoUrl() == "" {
			inputType := "本地视频"

			if videoInputContainer != nil && len(videoInputContainer.Objects) > 0 {
				for i := 0; i < len(videoInputContainer.Objects); i++ {
					// 如果对象是Container，查找其中的RadioGroup
					if container, ok := videoInputContainer.Objects[i].(*fyne.Container); ok {
						for j := 0; j < len(container.Objects); j++ {
							if radio, ok := container.Objects[j].(*widget.RadioGroup); ok {
								inputType = radio.Selected
								break
							}
						}
					}
				}
			}

			if inputType == "本地视频" {
				dialog.ShowError(fmt.Errorf("请先选择视频文件"), window)
			} else {
				dialog.ShowError(fmt.Errorf("请输入视频链接"), window)
			}
			progress.Hide()
			return
		}

		err := config.CheckConfig()
		if err != nil {
			dialog.ShowError(fmt.Errorf("配置不正确: %v", err), window)
			log.GetLogger().Error("配置不正确", zap.Error(err))
			progress.Hide()
			return
		}

		err = deps.CheckDependency()
		if err != nil {
			dialog.ShowError(fmt.Errorf("依赖环境准备失败: %v", err), window)
			log.GetLogger().Error("依赖环境准备失败", zap.Error(err))
			progress.Hide()
			return
		}
		btn.Hide()

		if config.ConfigBackup != config.Conf {
			if err = server.StopBackend(); err != nil {
				dialog.ShowError(fmt.Errorf("停止后端服务失败: %v", err), window)
				log.GetLogger().Error("停止后端服务失败", zap.Error(err))
				progress.Hide()
				return
			}

			go func() {
				err := server.StartBackend()
				if err != nil {
					dialog.ShowError(fmt.Errorf("启动后端服务失败: %v", err), window)
					log.GetLogger().Error("启动后端服务失败", zap.Error(err))
					progress.Hide()
					return
				}
			}()

			time.Sleep(1 * time.Second)
			config.ConfigBackup = config.Conf
		}

		if err = sm.StartTask(); err != nil {
			dialog.ShowError(err, window)
			progress.Hide()
			return
		}

		go func() {
			for {
				time.Sleep(1 * time.Second)
				if sm.progressBar.Value < 1 {
					continue
				}
				time.Sleep(1 * time.Second)
				if sm.progressBar.Value < 1 {
					continue
				}
				break
			}
			btn.Show()
			downloadContainer.Show()
		}()
		sm.progressBar.Refresh()
	}

	return btn
}
