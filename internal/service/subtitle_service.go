package service

import (
	"context"
	"errors"
	"fmt"
	"github.com/samber/lo"
	"go.uber.org/zap"
	"krillin-ai/internal/dto"
	"krillin-ai/internal/storage"
	"krillin-ai/internal/types"
	"krillin-ai/log"
	"krillin-ai/pkg/util"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func (s Service) StartSubtitleTask(req dto.StartVideoSubtitleTaskReq) (*dto.StartVideoSubtitleTaskResData, error) {
	// 校验链接
	if strings.Contains(req.Url, "youtube.com") {
		videoId, _ := util.GetYouTubeID(req.Url)
		if videoId == "" {
			return nil, fmt.Errorf("链接不合法")
		}
	}
	if strings.Contains(req.Url, "bilibili.com") {
		videoId := util.GetBilibiliVideoId(req.Url)
		if videoId == "" {
			return nil, fmt.Errorf("链接不合法")
		}
	}
	// 生成任务id
	taskId := util.GenerateRandStringWithUpperLowerNum(8)
	// 构造任务所需参数
	var resultType types.SubtitleResultType
	// 根据入参选项确定要返回的字幕类型
	if req.TargetLang == "none" {
		resultType = types.SubtitleResultTypeOriginOnly
	} else {
		if req.Bilingual == types.SubtitleTaskBilingualYes {
			if req.TranslationSubtitlePos == types.SubtitleTaskTranslationSubtitlePosTop {
				resultType = types.SubtitleResultTypeBilingualTranslationOnTop
			} else {
				resultType = types.SubtitleResultTypeBilingualTranslationOnBottom
			}
		} else {
			resultType = types.SubtitleResultTypeTargetOnly
		}
	}
	// 文字替换map
	replaceWordsMap := make(map[string]string)
	if len(req.Replace) > 0 {
		for _, replace := range req.Replace {
			beforeAfter := strings.Split(replace, "|")
			if len(beforeAfter) == 2 {
				replaceWordsMap[beforeAfter[0]] = beforeAfter[1]
			} else {
				log.GetLogger().Info("generateAudioSubtitles replace param length err", zap.Any("replace", replace), zap.Any("taskId", taskId))
			}
		}
	}
	var err error
	ctx := context.Background()
	// 创建字幕任务文件夹
	taskBasePath := filepath.Join("./tasks", taskId)
	if _, err = os.Stat(taskBasePath); os.IsNotExist(err) {
		// 不存在则创建
		err = os.MkdirAll(filepath.Join(taskBasePath, "output"), os.ModePerm)
		if err != nil {
			log.GetLogger().Error("StartVideoSubtitleTask MkdirAll err", zap.Any("req", req), zap.Error(err))
		}
	}

	// 创建任务
	storage.SubtitleTasks[taskId] = &types.SubtitleTask{
		TaskId:   taskId,
		VideoSrc: req.Url,
		Status:   types.SubtitleTaskStatusProcessing,
	}
	var ttsVoiceCode string
	if req.TtsVoiceCode == types.SubtitleTaskTtsVoiceCodeLongyu {
		ttsVoiceCode = "longyu"
	} else {
		ttsVoiceCode = "longchen"
	}

	// 处理声音克隆源
	var voiceCloneAudioUrl string
	if req.TtsVoiceCloneSrcFileUrl != "" {
		localFileUrl := strings.TrimPrefix(req.TtsVoiceCloneSrcFileUrl, "local:")
		fileKey := util.GenerateRandStringWithUpperLowerNum(5) + filepath.Ext(localFileUrl) // 防止url encode的问题，这里统一处理
		err = s.OssClient.UploadFile(context.Background(), fileKey, localFileUrl, s.OssClient.Bucket)
		if err != nil {
			log.GetLogger().Error("StartVideoSubtitleTask UploadFile err", zap.Any("req", req), zap.Error(err))
			return nil, errors.New("上传声音克隆源失败")
		}
		voiceCloneAudioUrl = fmt.Sprintf("https://%s.oss-cn-shanghai.aliyuncs.com/%s", s.OssClient.Bucket, fileKey)
		log.GetLogger().Info("StartVideoSubtitleTask 上传声音克隆源成功", zap.Any("oss url", voiceCloneAudioUrl))
	}

	stepParam := types.SubtitleTaskStepParam{
		TaskId:                  taskId,
		TaskBasePath:            taskBasePath,
		Link:                    req.Url,
		SubtitleResultType:      resultType,
		EnableModalFilter:       req.ModalFilter == types.SubtitleTaskModalFilterYes,
		EnableTts:               req.Tts == types.SubtitleTaskTtsYes,
		TtsVoiceCode:            ttsVoiceCode,
		VoiceCloneAudioUrl:      voiceCloneAudioUrl,
		ReplaceWordsMap:         replaceWordsMap,
		OriginLanguage:          types.StandardLanguageName(req.OriginLanguage),
		TargetLanguage:          types.StandardLanguageName(req.TargetLang),
		UserUILanguage:          types.StandardLanguageName(req.Language),
		EmbedSubtitleVideoType:  req.EmbedSubtitleVideoType,
		VerticalVideoMajorTitle: req.VerticalMajorTitle,
		VerticalVideoMinorTitle: req.VerticalMinorTitle,
		MaxWordOneLine:          12, // 默认值
	}
	if req.OriginLanguageWordOneLine != 0 {
		stepParam.MaxWordOneLine = req.OriginLanguageWordOneLine
	}
	go func() {
		defer func() {
			if r := recover(); r != nil {
				const size = 64 << 10
				buf := make([]byte, size)
				buf = buf[:runtime.Stack(buf, false)]
				log.GetLogger().Error("autoVideoSubtitle panic", zap.Any("panic:", r), zap.Any("stack:", buf))
				storage.SubtitleTasks[taskId].Status = types.SubtitleTaskStatusFailed
			}
		}()
		// 新版流程：链接->本地音频文件->视频信息获取（若有）->本地字幕文件->语言合成->视频合成->字幕文件链接生成
		log.GetLogger().Info("video subtitle start task", zap.String("taskId", taskId))
		err = s.linkToFile(ctx, &stepParam)
		if err != nil {
			log.GetLogger().Error("StartVideoSubtitleTask linkToFile err", zap.Any("req", req), zap.Error(err))
			storage.SubtitleTasks[stepParam.TaskId].Status = types.SubtitleTaskStatusFailed
			storage.SubtitleTasks[stepParam.TaskId].FailReason = err.Error()
			return
		}
		// 暂时不加视频信息
		//err = s.getVideoInfo(ctx, &stepParam)
		//if err != nil {
		//	log.GetLogger().Error("StartVideoSubtitleTask getVideoInfo err", zap.Any("req", req), zap.Error(err))
		//	storage.SubtitleTasks[stepParam.TaskId].Status = types.SubtitleTaskStatusFailed
		//	storage.SubtitleTasks[stepParam.TaskId].FailReason = "get video info error"
		//	return
		//}
		err = s.audioToSubtitle(ctx, &stepParam)
		if err != nil {
			log.GetLogger().Error("StartVideoSubtitleTask audioToSubtitle err", zap.Any("req", req), zap.Error(err))
			storage.SubtitleTasks[stepParam.TaskId].Status = types.SubtitleTaskStatusFailed
			storage.SubtitleTasks[stepParam.TaskId].FailReason = err.Error()
			return
		}
		err = s.srtFileToSpeech(ctx, &stepParam)
		if err != nil {
			log.GetLogger().Error("StartVideoSubtitleTask srtFileToSpeech err", zap.Any("req", req), zap.Error(err))
			storage.SubtitleTasks[stepParam.TaskId].Status = types.SubtitleTaskStatusFailed
			storage.SubtitleTasks[stepParam.TaskId].FailReason = err.Error()
			return
		}
		err = s.embedSubtitles(ctx, &stepParam)
		if err != nil {
			log.GetLogger().Error("StartVideoSubtitleTask embedSubtitles err", zap.Any("req", req), zap.Error(err))
			storage.SubtitleTasks[stepParam.TaskId].Status = types.SubtitleTaskStatusFailed
			storage.SubtitleTasks[stepParam.TaskId].FailReason = err.Error()
			return
		}
		err = s.uploadSubtitles(ctx, &stepParam)
		if err != nil {
			log.GetLogger().Error("StartVideoSubtitleTask uploadSubtitles err", zap.Any("req", req), zap.Error(err))
			storage.SubtitleTasks[stepParam.TaskId].Status = types.SubtitleTaskStatusFailed
			storage.SubtitleTasks[stepParam.TaskId].FailReason = err.Error()
			return
		}

		log.GetLogger().Info("video subtitle task end", zap.String("taskId", taskId))
	}()

	return &dto.StartVideoSubtitleTaskResData{
		TaskId: taskId,
	}, nil
}

func (s Service) GetTaskStatus(req dto.GetVideoSubtitleTaskReq) (*dto.GetVideoSubtitleTaskResData, error) {
	task := storage.SubtitleTasks[req.TaskId]
	if task == nil {
		return nil, errors.New("任务不存在")
	}
	if task.Status == types.SubtitleTaskStatusFailed {
		return nil, fmt.Errorf("任务失败，原因：%s", task.FailReason)
	}
	return &dto.GetVideoSubtitleTaskResData{
		TaskId:         task.TaskId,
		ProcessPercent: task.ProcessPct,
		VideoInfo: &dto.VideoInfo{
			Title:                 task.Title,
			Description:           task.Description,
			TranslatedTitle:       task.TranslatedTitle,
			TranslatedDescription: task.TranslatedDescription,
		},
		SubtitleInfo: lo.Map(task.SubtitleInfos, func(item types.SubtitleInfo, _ int) *dto.SubtitleInfo {
			return &dto.SubtitleInfo{
				Name:        item.Name,
				DownloadUrl: item.DownloadUrl,
			}
		}),
		TargetLanguage:    task.TargetLanguage,
		SpeechDownloadUrl: task.SpeechDownloadUrl,
	}, nil
}
