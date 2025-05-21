package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"krillin-ai/config"
	"krillin-ai/log"

	openai "github.com/sashabaranov/go-openai"
	"go.uber.org/zap"
)

func (c *Client) ChatCompletion(query string) (string, error) {
	var responseFormat *openai.ChatCompletionResponseFormat

	if config.Conf.Openai.JsonLLM {
		responseFormat = &openai.ChatCompletionResponseFormat{
			Type: "json_schema",
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name:   "translation_response",
				Strict: true,
				Schema: json.RawMessage(`{
					"type": "array",
					"items": {
						"type": "object",
						"properties": {
							"original_sentence": {"type": "string"},
							"translated_sentence": {"type": "string"}
						},
						"required": ["original_sentence", "translated_sentence"]
					}
				}`),
			},
		}
	} else {
		responseFormat = &openai.ChatCompletionResponseFormat{
			Type: "text",
		}
	}

	req := openai.ChatCompletionRequest{
		Model: openai.GPT4oMini20240718,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "You are an assistant that helps with subtitle translation.",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: query,
			},
		},
		Temperature:    0.9,
		Stream:         config.Conf.Openai.Stream,
		MaxTokens:      8192,
		ResponseFormat: responseFormat,
	}

	if config.Conf.Openai.Model != "" {
		req.Model = config.Conf.Openai.Model
	}

	var resContent string

	if !config.Conf.Openai.Stream {
		// Use non-streaming mode for JSON responses
		resp, err := c.client.CreateChatCompletion(context.Background(), req)
		if err != nil {
			log.GetLogger().Error("openai create chat completion failed", zap.Error(err))
			return "", err
		}

		if len(resp.Choices) == 0 {
			log.GetLogger().Info("openai response has no choices")
			return "", fmt.Errorf("no response choices available")
		}

		resContent = resp.Choices[0].Message.Content
	} else {
		// Use streaming mode for text responses
		stream, err := c.client.CreateChatCompletionStream(context.Background(), req)
		if err != nil {
			log.GetLogger().Error("openai create chat completion stream failed", zap.Error(err))
			return "", err
		}
		defer stream.Close()

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
	}

	if config.Conf.Openai.JsonLLM {
		parsedContent, err := parseJSONResponse(resContent)
		if err != nil {
			log.GetLogger().Error("failed to parse JSON response", zap.Error(err))
			return "", err
		}
		return parsedContent, nil
	}

	return resContent, nil
}

func parseJSONResponse(jsonStr string) (string, error) {
	var jsonData []map[string]string
	err := json.Unmarshal([]byte(jsonStr), &jsonData)
	if err != nil {
		return "", err
	}

	var result string
	for i, item := range jsonData {
		result += fmt.Sprintf("%d\n%s\n%s\n\n", i+1, item["translated_sentence"], item["original_sentence"])
	}

	return result, nil
}
