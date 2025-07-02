package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"krillin-ai/config"
	"krillin-ai/log"
	"net/http"
	"os"
	"strings"

	openai "github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
)

func (c *Client) ChatCompletion(query string) (string, error) {
	var responseFormat *openai.ChatCompletionResponseFormat
	var systemMessage string

	if config.Conf.Llm.Json {
		responseFormat = &openai.ChatCompletionResponseFormat{
			Type: "json_schema",
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name:   "translation_response",
				Strict: true,
				Schema: json.RawMessage(`{
					"type": "object",
					"properties": {
						"translations": {
							"type": "array",
							"items": {
								"type": "object",
								"properties": {
									"original_sentence": {"type": "string"},
									"translated_sentence": {"type": "string"}
								},
								"required": ["original_sentence", "translated_sentence"],
								"additionalProperties": false
							}
						}
					},
					"required": ["translations"],
					"additionalProperties": false
				}`),
			},
		}
		systemMessage = "You are an assistant that helps with subtitle translation. Please return your response in JSON format according to the specified schema."
	} else {
		responseFormat = &openai.ChatCompletionResponseFormat{
			Type: "text",
		}
		systemMessage = "You are an assistant that helps with subtitle translation."
	}

	req := openai.ChatCompletionRequest{
		Model: config.Conf.Llm.Model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemMessage,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: query,
			},
		},
		Temperature:    0.9,
		Stream:         true,
		MaxTokens:      8192,
		ResponseFormat: responseFormat,
	}

	stream, err := c.client.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		log.GetLogger().Error("openai create chat completion stream failed", zap.Error(err))
		return "", err
	}
	defer stream.Close()

	var resContent string
	for {
		response, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.GetLogger().Error("openai stream receive failed", zap.Error(err))
			return "", err
		}
		if len(response.Choices) == 0 {
			log.GetLogger().Info("openai stream receive no choices", zap.Any("response", response))
			continue
		}

		resContent += response.Choices[0].Delta.Content
	}

	if config.Conf.Llm.Json {
		parsedContent, err := parseJSONResponse(resContent)
		if err != nil {
			log.GetLogger().Error("failed to parse JSON response", zap.Error(err))
			return "", err
		}
		return parsedContent, nil
	}

	return resContent, nil
}

// ChatCompletionSingle 专门用于单个翻译的聊天完成，使用单个翻译对象的JSON Schema
func (c *Client) ChatCompletionSingle(query string) (string, error) {
	var responseFormat *openai.ChatCompletionResponseFormat
	var systemMessage string

	if config.Conf.Llm.Json {
		responseFormat = &openai.ChatCompletionResponseFormat{
			Type: "json_schema",
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name:   "single_translation_response",
				Strict: true,
				Schema: json.RawMessage(`{
					"type": "object",
					"properties": {
						"original_sentence": {"type": "string"},
						"translated_sentence": {"type": "string"}
					},
					"required": ["original_sentence", "translated_sentence"],
					"additionalProperties": false
				}`),
			},
		}
		systemMessage = "You are an assistant that helps with subtitle translation. Please return your response in JSON format according to the specified schema."
	} else {
		responseFormat = &openai.ChatCompletionResponseFormat{
			Type: "text",
		}
		systemMessage = "You are an assistant that helps with subtitle translation."
	}

	req := openai.ChatCompletionRequest{
		Model: config.Conf.Llm.Model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemMessage,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: query,
			},
		},
		Temperature:    0.9,
		Stream:         true,
		MaxTokens:      8192,
		ResponseFormat: responseFormat,
	}

	stream, err := c.client.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		log.GetLogger().Error("openai create chat completion stream failed", zap.Error(err))
		return "", err
	}
	defer stream.Close()

	var resContent string
	for {
		response, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.GetLogger().Error("openai stream receive failed", zap.Error(err))
			return "", err
		}
		if len(response.Choices) == 0 {
			log.GetLogger().Info("openai stream receive no choices", zap.Any("response", response))
			continue
		}

		resContent += response.Choices[0].Delta.Content
	}

	if config.Conf.Llm.Json {
		// 直接返回JSON字符串，让调用方解析
		return resContent, nil
	}

	return resContent, nil
}

// ChatCompletionSplitLong 专门用于长句分割的聊天完成，使用分割对象的JSON Schema
func (c *Client) ChatCompletionSplitLong(query string) (string, error) {
	var responseFormat *openai.ChatCompletionResponseFormat
	var systemMessage string

	if config.Conf.Llm.Json {
		responseFormat = &openai.ChatCompletionResponseFormat{
			Type: "json_schema",
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name:   "split_long_sentence_response",
				Strict: true,
				Schema: json.RawMessage(`{
					"type": "object",
					"properties": {
						"align": {
							"type": "array",
							"items": {
								"type": "object",
								"properties": {
									"origin_part": {"type": "string"},
									"translated_part": {"type": "string"}
								},
								"required": ["origin_part", "translated_part"],
								"additionalProperties": false
							}
						}
					},
					"required": ["align"],
					"additionalProperties": false
				}`),
			},
		}
		systemMessage = "You are an assistant that helps with subtitle translation. Please return your response in JSON format according to the specified schema."
	} else {
		responseFormat = &openai.ChatCompletionResponseFormat{
			Type: "text",
		}
		systemMessage = "You are an assistant that helps with subtitle translation."
	}

	req := openai.ChatCompletionRequest{
		Model: config.Conf.Llm.Model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemMessage,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: query,
			},
		},
		Temperature:    0.9,
		Stream:         true,
		MaxTokens:      8192,
		ResponseFormat: responseFormat,
	}

	stream, err := c.client.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		log.GetLogger().Error("openai create chat completion stream failed", zap.Error(err))
		return "", err
	}
	defer stream.Close()

	var resContent string
	for {
		response, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.GetLogger().Error("openai stream receive failed", zap.Error(err))
			return "", err
		}
		if len(response.Choices) == 0 {
			log.GetLogger().Info("openai stream receive no choices", zap.Any("response", response))
			continue
		}

		resContent += response.Choices[0].Delta.Content
	}

	if config.Conf.Llm.Json {
		// 直接返回JSON字符串，让调用方解析
		return resContent, nil
	}

	return resContent, nil
}

func (c *Client) Text2Speech(text, voice string, outputFile string) error {
	baseUrl := config.Conf.Tts.Openai.BaseUrl
	if baseUrl == "" {
		baseUrl = "https://api.openai.com/v1"
	}
	url := baseUrl + "/audio/speech"

	// 创建HTTP请求
	reqBody := fmt.Sprintf(`{
		"model": "tts-1",
		"input": "%s",
		"voice":"%s",
		"response_format": "wav"
	}`, text, voice)
	req, err := http.NewRequest("POST", url, strings.NewReader(reqBody))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", config.Conf.Tts.Openai.ApiKey))

	// 发送HTTP请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.GetLogger().Error("openai tts failed", zap.Int("status_code", resp.StatusCode), zap.String("body", string(body)))
		return fmt.Errorf("openai tts none-200 status code: %d", resp.StatusCode)
	}

	file, err := os.Create(outputFile)
	if err != nil {
		return err
	}
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	if err != nil {
		return err
	}

	return nil
}

func parseJSONResponse(jsonStr string) (string, error) {
	// 清理可能的markdown代码块格式
	jsonStr = strings.TrimSpace(jsonStr)

	// 移除可能的 ```json 开头和 ``` 结尾
	if strings.HasPrefix(jsonStr, "```json") {
		jsonStr = strings.TrimPrefix(jsonStr, "```json")
		jsonStr = strings.TrimSpace(jsonStr)
	} else if strings.HasPrefix(jsonStr, "```") {
		jsonStr = strings.TrimPrefix(jsonStr, "```")
		jsonStr = strings.TrimSpace(jsonStr)
	}

	if strings.HasSuffix(jsonStr, "```") {
		jsonStr = strings.TrimSuffix(jsonStr, "```")
		jsonStr = strings.TrimSpace(jsonStr)
	}

	// 记录清理后的JSON字符串用于调试
	log.GetLogger().Debug("cleaned JSON response", zap.String("json", jsonStr))

	var response struct {
		Translations []struct {
			Original   string `json:"original_sentence"`
			Translated string `json:"translated_sentence"`
		} `json:"translations"`
	}

	err := json.Unmarshal([]byte(jsonStr), &response)
	if err != nil {
		// 记录原始响应以便调试
		log.GetLogger().Error("failed to parse JSON after cleanup",
			zap.Error(err),
			zap.String("original_response", jsonStr))
		return "", fmt.Errorf("failed to parse JSON: %v", err)
	}

	var result strings.Builder
	for i, item := range response.Translations {
		result.WriteString(fmt.Sprintf("%d\n%s\n%s\n\n",
			i+1,
			item.Translated,
			item.Original))
	}

	return result.String(), nil
}
