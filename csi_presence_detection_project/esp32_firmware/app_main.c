#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"

#include "nvs_flash.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"

#include "ping/ping_sock.h"
#include "protocol_examples_common.h" // For example_connect

#include "mqtt_client.h" // MQTT Library

static const char *TAG = "csi_mqtt_sender";

// --- MQTT Broker Settings ---
#define MQTT_BROKER_URL "mqtt://192.168.1.11" // <<< 【務必修改為您的MQTT Broker IP】
#define MQTT_PUBLISH_TOPIC "homebridge/csi/raw_data" // <<< 務必與 Homebridge index.js 中的 mqttCsiTopic 一致

// --- CSI Data Structure (for queue) ---
#define MAX_CSI_BUF_LEN 128
typedef struct {
    int8_t csi_buf[MAX_CSI_BUF_LEN];
    int csi_len;
    int rssi;
    int mcs;
} csi_data_packet_t;

// --- FreeRTOS Queue ---
#define CSI_QUEUE_LENGTH 10 // 增加隊列長度以緩衝因速率控制可能積壓的包
#define CSI_QUEUE_ITEM_SIZE sizeof(csi_data_packet_t)
static QueueHandle_t csi_data_queue;

// --- Global MQTT Client Handle & Event Group ---
esp_mqtt_client_handle_t mqtt_client = NULL;
static EventGroupHandle_t mqtt_event_group;
const int MQTT_CONNECTED_BIT = BIT0;

// Simplified rx_ctrl structure
typedef struct {
    signed rssi : 8;
    unsigned rate : 5;
    unsigned : 1;
    unsigned sig_mode : 2;
    unsigned : 6;
    unsigned mcs : 7;
} wifi_pkt_rx_ctrl_t_simple;


// CSI Reception Callback
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
    // ***** MODIFICATION: 恢復了 !ctx 的檢查 *****
    if (!info || !info->buf || !ctx || info->len == 0) {
        // ESP_LOGW(TAG, "wifi_csi_cb: Invalid arguments or zero length CSI");
        return;
    }
    // MAC address filtering (ctx is s_ap_info.bssid from wifi_csi_init)
    if (memcmp(info->mac, ctx, 6)) { // 確保這行是有效的，如果 ctx 是 NULL 則不應執行
         return;
    }
    // ***** END MODIFICATION *****

    if (info->len > MAX_CSI_BUF_LEN) {
        ESP_LOGE(TAG, "Received CSI data length (%d) > MAX_CSI_BUF_LEN (%d). Discarding.", info->len, MAX_CSI_BUF_LEN);
        return;
    }

    csi_data_packet_t packet;
    packet.rssi = info->rx_ctrl.rssi;
    packet.mcs = info->rx_ctrl.mcs;
    packet.csi_len = info->len;
    memcpy(packet.csi_buf, info->buf, info->len);

    if (xQueueSend(csi_data_queue, &packet, (TickType_t)0) != pdPASS) {
        ESP_LOGW(TAG, "CSI data queue full. Packet discarded."); // 建議取消註釋此行以便調試
    }
}

// MQTT Event Handler Callback (與您提供的一致)
static void mqtt_event_handler_cb(esp_mqtt_event_handle_t event) {
    switch (event->event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED to %s", MQTT_BROKER_URL);
            xEventGroupSetBits(mqtt_event_group, MQTT_CONNECTED_BIT);
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "MQTT_EVENT_DISCONNECTED");
            xEventGroupClearBits(mqtt_event_group, MQTT_CONNECTED_BIT);
            break;
        case MQTT_EVENT_SUBSCRIBED:
            ESP_LOGI(TAG, "MQTT_EVENT_SUBSCRIBED, msg_id=%d", event->msg_id);
            break;
        case MQTT_EVENT_UNSUBSCRIBED:
            ESP_LOGI(TAG, "MQTT_EVENT_UNSUBSCRIBED, msg_id=%d", event->msg_id);
            break;
        case MQTT_EVENT_PUBLISHED:
            ESP_LOGD(TAG, "MQTT_EVENT_PUBLISHED, msg_id=%d", event->msg_id);
            break;
        case MQTT_EVENT_DATA:
            ESP_LOGI(TAG, "MQTT_EVENT_DATA");
            break;
        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT_EVENT_ERROR");
            if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                ESP_LOGE(TAG, "Last error code reported from esp-tls: 0x%x", event->error_handle->esp_tls_last_esp_err);
                ESP_LOGE(TAG, "Last error code reported from tls stack: 0x%x", event->error_handle->esp_tls_stack_err);
            } else if (event->error_handle->error_type == MQTT_ERROR_TYPE_CONNECTION_REFUSED) {
                ESP_LOGE(TAG, "Connection refused error: 0x%x", event->error_handle->connect_return_code);
            } else {
                ESP_LOGW(TAG, "Unknown error type: 0x%x", event->error_handle->error_type);
            }
            break;
        default:
            ESP_LOGD(TAG, "Other event id:%d", event->event_id);
            break;
    }
}

// MQTT Event Handler wrapper (與您提供的一致)
static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    ESP_LOGD(TAG, "Event dispatched from event loop base=%s, event_id=%ld", base, (long)event_id);
    mqtt_event_handler_cb(event_data);
}


// MQTT Sending Task
static void mqtt_send_task(void *pvParameters) {
    csi_data_packet_t received_packet;
    static char json_buffer[1024]; 
    static char csi_raw_str_buffer[700];

    // ***** MODIFICATION: 新增用於速率控制的變量 *****
    static uint32_t last_publish_time_ms = 0;
    const uint32_t PUBLISH_INTERVAL_MS = 1000 / 17; // 目標16Hz，約62.5ms
    // ***** END MODIFICATION *****

    ESP_LOGI(TAG, "MQTT Send Task started. Target publish interval: %lu ms (~16Hz). Waiting for MQTT connection and data from queue.", (unsigned long)PUBLISH_INTERVAL_MS);


    while (1) {
        EventBits_t bits = xEventGroupWaitBits(mqtt_event_group, MQTT_CONNECTED_BIT,
                                               pdFALSE, pdTRUE, portMAX_DELAY);

        if (!(bits & MQTT_CONNECTED_BIT)) {
            ESP_LOGW(TAG, "MQTT not connected. Retrying or waiting...");
            vTaskDelay(pdMS_TO_TICKS(1000)); 
            continue;
        }

        // 從隊列中獲取CSI數據包
        if (xQueueReceive(csi_data_queue, &received_packet, pdMS_TO_TICKS(PUBLISH_INTERVAL_MS > 100 ? 100 : PUBLISH_INTERVAL_MS / 2)) == pdPASS) {
        // 使用帶有超時的 xQueueReceive，以允許定期檢查時間，而不是永遠阻塞
        // 如果沒有數據，它會在超時後返回，然後可以檢查是否到了發送時間（如果之前有數據但未到時間）
        // 或者如果收到數據，則進入速率控制邏輯

            if (received_packet.csi_len != 128) { 
                ESP_LOGW(TAG, "MQTT Task: Received packet with unexpected CSI length: %d. Skipping.", received_packet.csi_len);
                continue;
            }

            // ***** MODIFICATION: 加入速率控制邏輯 *****
            uint32_t current_time_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
            if (last_publish_time_ms == 0 || (current_time_ms - last_publish_time_ms) >= PUBLISH_INTERVAL_MS) {
                
                // ----- 開始構建 csi_raw 字符串 -----
                csi_raw_str_buffer[0] = '[';
                int offset = 1;
                for (int i = 0; i < received_packet.csi_len; i++) {
                    offset += snprintf(csi_raw_str_buffer + offset,
                                     sizeof(csi_raw_str_buffer) - offset,
                                     "%d%s",
                                     received_packet.csi_buf[i],
                                     (i < received_packet.csi_len - 1) ? "," : "");
                    if (offset >= sizeof(csi_raw_str_buffer) - 2) { 
                        ESP_LOGE(TAG, "CSI raw string buffer overflow in mqtt_send_task!");
                        offset = -1; 
                        break;
                    }
                }

                if (offset == -1 || offset >= sizeof(csi_raw_str_buffer) - 1) { 
                    ESP_LOGE(TAG, "Failed to build csi_raw_str_buffer or buffer was full in mqtt_send_task.");
                    continue; 
                }
                csi_raw_str_buffer[offset++] = ']';
                csi_raw_str_buffer[offset] = '\0';
                // ----- csi_raw 字符串構建完畢 -----

                // ----- 構建完整的 JSON 負載 -----
                snprintf(json_buffer, sizeof(json_buffer),
                         "{\"csi_raw\": \"%s\", \"rssi\": %d, \"mcs\": %d}",
                         csi_raw_str_buffer, 
                         received_packet.rssi,
                         received_packet.mcs);
                // ----- JSON 負載構建完畢 -----

                int msg_id = esp_mqtt_client_publish(mqtt_client, MQTT_PUBLISH_TOPIC, json_buffer, 0, 0, 0);
                if (msg_id >= 0) { 
                    last_publish_time_ms = current_time_ms; // 更新上次成功發布的時間
                    ESP_LOGD(TAG, "MQTT publish initiated, msg_id=%d, rate_limited to ~%uHz", msg_id, (unsigned int)(1000/PUBLISH_INTERVAL_MS));
                } else {
                    ESP_LOGE(TAG, "MQTT publish failed, error: %d. (MQTT client might be disconnected)", msg_id);
                }
            } else {
                // 如果未達到發送間隔，則丟棄此從隊列中取出的包
                ESP_LOGD(TAG, "Packet from queue discarded, %lu ms remaining for next publish slot.", (unsigned long)(PUBLISH_INTERVAL_MS - (current_time_ms - last_publish_time_ms)));
            }
            // ***** END MODIFICATION *****
        } else {
            // xQueueReceive 超時，沒有從隊列中收到數據包
            // 這可以讓循環繼續，並在下一次檢查 MQTT 連接狀態和時間
            // 同時，如果 last_publish_time_ms != 0 且 (current_time_ms - last_publish_time_ms) >= PUBLISH_INTERVAL_MS
            // 但隊列為空，說明CSI產生速率低於目標速率，這種情況下我們等待下一個包即可。
            // ESP_LOGD(TAG, "No packet in queue within timeout.");
        }
    }
}

// Initialize and start MQTT client (與您提供的一致)
static void mqtt_app_start(void) {
    mqtt_event_group = xEventGroupCreate();

    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = MQTT_BROKER_URL,
    };

    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    if (mqtt_client == NULL) {
        ESP_LOGE(TAG, "Failed to init MQTT client");
        return;
    }
    esp_err_t err = esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to register MQTT event handler: %s", esp_err_to_name(err));
        return;
    }
    err = esp_mqtt_client_start(mqtt_client);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start MQTT client: %s", esp_err_to_name(err));
    } else {
        ESP_LOGI(TAG, "MQTT client started successfully. Waiting for connection to broker...");
    }
}

// wifi_csi_init (與您提供的主要一致，確保 ctx 傳遞)
static void wifi_csi_init(void) {
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = false,
        .stbc_htltf2_en = false,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false, 
        .shift = 0,    
    };

    static wifi_ap_record_t s_ap_info = {0};
    esp_err_t ret = esp_wifi_sta_get_ap_info(&s_ap_info);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to get AP info: %s. CSI will listen to all MACs (ctx=NULL).", esp_err_to_name(ret));
        ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
        ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL)); // 傳遞 NULL 作為 ctx
    } else {
        ESP_LOGI(TAG, "CSI will monitor MAC: " MACSTR, MAC2STR(s_ap_info.bssid));
        ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
        ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, s_ap_info.bssid)); // 傳遞 AP BSSID 作為 ctx
    }

    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    ESP_LOGI(TAG, "CSI reception initialized.");
}

// ***** MODIFICATION: 調整 PING_FREQUENCY_HZ *****
#define PING_FREQUENCY_HZ 100 // 嘗試 20Hz 或 30Hz，以確保有足夠的CSI包供速率控制器選擇
// ***** END MODIFICATION *****

static esp_err_t wifi_ping_router_start(void) {
    static esp_ping_handle_t ping_handle = NULL;
    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();

    ping_config.interval_ms = 1000 / PING_FREQUENCY_HZ;
    ping_config.count = 0; 
    ping_config.task_stack_size = 3072; 
    ping_config.data_size = 1;    

    esp_netif_ip_info_t local_ip;
    esp_netif_t *sta_netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (sta_netif) {
        if (esp_netif_get_ip_info(sta_netif, &local_ip) == ESP_OK) {
            ESP_LOGI(TAG, "Local IP: " IPSTR ", Gateway: " IPSTR, IP2STR(&local_ip.ip), IP2STR(&local_ip.gw));
            ping_config.target_addr.u_addr.ip4.addr = ip4_addr_get_u32(&local_ip.gw); 
            ping_config.target_addr.type = ESP_IPADDR_TYPE_V4;
            esp_ping_callbacks_t cbs = {0}; 
            esp_ping_new_session(&ping_config, &cbs, &ping_handle);
            esp_ping_start(ping_handle);
            ESP_LOGI(TAG, "Ping to router started (Target Ping Frequency: %d Hz).", PING_FREQUENCY_HZ);
        } else {
            ESP_LOGE(TAG, "Failed to get IP info for STA netif for ping.");
            return ESP_FAIL;
        }
    } else {
        ESP_LOGE(TAG, "Failed to get STA netif handle for ping.");
        return ESP_FAIL;
    }
    return ESP_OK;
}

void app_main() {
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    ESP_LOGI(TAG, "Connecting to WiFi...");
    ESP_ERROR_CHECK(example_connect()); 
    ESP_LOGI(TAG, "WiFi Connected.");

    mqtt_app_start(); 

    csi_data_queue = xQueueCreate(CSI_QUEUE_LENGTH, CSI_QUEUE_ITEM_SIZE);
    if (csi_data_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create CSI data queue. Halting.");
        if (mqtt_client) { esp_mqtt_client_stop(mqtt_client); esp_mqtt_client_destroy(mqtt_client); }
        if (mqtt_event_group) vEventGroupDelete(mqtt_event_group);
        return;
    }
    ESP_LOGI(TAG, "CSI data queue created.");

    if (xTaskCreate(mqtt_send_task, "mqtt_send_task", 4096, NULL, 5, NULL) != pdPASS) {
        ESP_LOGE(TAG, "Failed to create MQTT send task. Halting.");
        vQueueDelete(csi_data_queue); 
        if (mqtt_client) { esp_mqtt_client_stop(mqtt_client); esp_mqtt_client_destroy(mqtt_client); }
        if (mqtt_event_group) vEventGroupDelete(mqtt_event_group);
        return;
    }

    wifi_csi_init();
    wifi_ping_router_start(); 

    ESP_LOGI(TAG, "ESP32 CSI MQTT Sender initialized. Broker: %s, Topic: %s", MQTT_BROKER_URL, MQTT_PUBLISH_TOPIC);
    ESP_LOGI(TAG, "Monitoring free heap. Initial free: %lu bytes", esp_get_free_heap_size());
}