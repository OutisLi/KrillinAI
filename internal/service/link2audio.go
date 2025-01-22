package service

import (
	"context"
	"errors"
	"fmt"
	"go.uber.org/zap"
	"krillin-ai/internal/storage"
	"krillin-ai/internal/types"
	"krillin-ai/log"
	"krillin-ai/pkg/util"
	"os/exec"
	"strings"
)

func (s Service) linkToFile(ctx context.Context, stepParam *types.SubtitleTaskStepParam) error {
	var (
		err    error
		output []byte
	)
	link := stepParam.Link
	audioPath := fmt.Sprintf("%s/%s", stepParam.TaskBasePath, types.SubtitleTaskAudioFileName)
	videoPath := fmt.Sprintf("%s/%s", stepParam.TaskBasePath, types.SubtitleTaskVideoFileName)
	storage.SubtitleTasks[stepParam.TaskId].ProcessPct = 3
	if strings.Contains(link, "local:") {
		// 本地文件
		videoPath = strings.ReplaceAll(link, "local:", "")
		stepParam.InputVideoPath = videoPath
		cmd := exec.Command(storage.FfmpegPath, "-i", videoPath, "-vn", "-ar", "44100", "-ac", "2", "-ab", "192k", "-f", "mp3", audioPath)
		output, err = cmd.CombinedOutput()
		if err != nil {
			log.GetLogger().Error("generateAudioSubtitles.linkToFile ffmpeg error", zap.Any("step param", stepParam), zap.String("output", string(output)), zap.Error(err))
			return fmt.Errorf("generateAudioSubtitles.linkToFile ffmpeg error: %w", err)
		}
	} else if strings.Contains(link, "youtube.com") {
		var videoId string
		videoId, err = util.GetYouTubeID(link)
		if err != nil {
			log.GetLogger().Error("linkToFile.GetYouTubeID error", zap.Any("step param", stepParam), zap.Error(err))
			return fmt.Errorf("linkToFile.GetYouTubeID error: %w", err)
		}
		stepParam.Link = "https://www.youtube.com/watch?v=" + videoId
		cmdArgs := []string{"-f", "bestaudio", "--extract-audio", "--audio-format", "mp3", "--audio-quality", "192K", "-o", audioPath, stepParam.Link}

		cmdArgs = append(cmdArgs, "--cookies", "./cookies.txt")
		if storage.FfmpegPath != "ffmpeg" {
			cmdArgs = append(cmdArgs, "--ffmpeg-location", storage.FfmpegPath)
		}
		cmd := exec.Command(storage.YtdlpPath, cmdArgs...)
		output, err = cmd.CombinedOutput()
		if err != nil {
			log.GetLogger().Error("linkToFile download audio yt-dlp error", zap.Any("step param", stepParam), zap.String("output", string(output)), zap.Error(err))
			return fmt.Errorf("linkToFile download audio yt-dlp error: %w", err)
		}
	} else if strings.Contains(link, "bilibili.com") {
		videoId := util.GetBilibiliVideoId(link)
		if videoId == "" {
			return errors.New("linkToFile error: invalid link")
		}
		stepParam.Link = "https://www.bilibili.com/video/" + videoId
		cmdArgs := []string{"-f", "bestaudio[ext=m4a]", "-x", "--audio-format", "mp3", "-o", audioPath, stepParam.Link}
		if storage.FfmpegPath != "ffmpeg" {
			cmdArgs = append(cmdArgs, "--ffmpeg-location", storage.FfmpegPath)
		}
		cmd := exec.Command(storage.YtdlpPath, cmdArgs...)
		output, err = cmd.CombinedOutput()
		if err != nil {
			log.GetLogger().Error("linkToFile download audio yt-dlp error", zap.Any("step param", stepParam), zap.String("output", string(output)), zap.Error(err))
			return fmt.Errorf("linkToFile download audio yt-dlp error: %w", err)
		}
	} else {
		log.GetLogger().Info("linkToFile.unsupported link type", zap.Any("step param", stepParam))
		return errors.New("linkToFile error: unsupported link, only support youtube, bilibili and local file")
	}
	storage.SubtitleTasks[stepParam.TaskId].ProcessPct = 6
	stepParam.AudioFilePath = audioPath

	if !strings.HasPrefix(link, "local:") && stepParam.EmbedSubtitleVideoType != "none" {
		// 需要下载原视频
		cmd := exec.Command(storage.YtdlpPath, "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]", "-o", videoPath, stepParam.Link)
		output, err = cmd.CombinedOutput()
		if err != nil {
			log.GetLogger().Error("linkToFile download video yt-dlp error", zap.Any("step param", stepParam), zap.String("output", string(output)), zap.Error(err))
			return fmt.Errorf("linkToFile download video yt-dlp error: %w", err)
		}
		stepParam.InputVideoPath = videoPath
	}

	// 更新字幕任务信息
	storage.SubtitleTasks[stepParam.TaskId].ProcessPct = 10
	return nil
}
