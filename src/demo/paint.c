#include "raylib.h"
#include <math.h>
#define CANVAS_SIZE 280
#define MNIST_SIZE 28
#define BRUSH_RADIUS 14.0f

int NeuralNet_Predict(float input[784])
{
    float sum = 0.0f;
    for (int i = 0; i < 784; i++) sum += input[i];
    if (sum < 10.0f) return -1;
    return (int)fmod(sum, 10);
}

int main(void)
{
    InitWindow(800, 500, "MNIST Digit Input");
    SetTargetFPS(60);

    RenderTexture2D drawCanvas = LoadRenderTexture(CANVAS_SIZE, CANVAS_SIZE);
    SetTextureFilter(drawCanvas.texture, TEXTURE_FILTER_BILINEAR); // blend pixels smoothly
    RenderTexture2D mnistCanvas = LoadRenderTexture(MNIST_SIZE, MNIST_SIZE);

    BeginTextureMode(drawCanvas);
    ClearBackground(BLACK);
    EndTextureMode();

    BeginTextureMode(mnistCanvas);
    ClearBackground(BLACK);
    EndTextureMode();

    Vector2 prev = { -1, -1 };
    bool drawing = false;
    int predictedDigit = -1;

    float nnInput[784] = {0};

    while (!WindowShouldClose())
    {
        Vector2 mouse = GetMousePosition();
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            if (mouse.x >= 20 && mouse.x <= 20 + CANVAS_SIZE &&
                mouse.y >= 80 && mouse.y <= 80 + CANVAS_SIZE) {
                Vector2 p = { mouse.x - 20, mouse.y - 80 };
                BeginTextureMode(drawCanvas);
                if (drawing) {
                    float dx = p.x - prev.x;
                    float dy = p.y - prev.y;
                    float dist = sqrtf(dx*dx + dy*dy);
                    int steps = (int)(dist / (BRUSH_RADIUS * 0.5f));
                    for (int i = 0; i <= steps; i++) {
                        float t =  (float)i / steps ;
                        Vector2 pos = {
                            prev.x + dx * t,
                            prev.y + dy * t
                        };
                        DrawCircleV(pos, BRUSH_RADIUS, WHITE);
                    }
                }

                DrawCircleV(p, BRUSH_RADIUS, WHITE);
                prev = p;
                drawing = true;

                EndTextureMode();
            }
        } else {
            drawing = false;
        }

        BeginTextureMode(mnistCanvas);
            ClearBackground(BLACK);
            DrawTexturePro(
                drawCanvas.texture,
                (Rectangle){ 0, 0, CANVAS_SIZE, CANVAS_SIZE },
                (Rectangle){ 0, 0, MNIST_SIZE, MNIST_SIZE },
                (Vector2){ 0, 0 },
                0,
                WHITE
            );
        EndTextureMode();

        Image img = LoadImageFromTexture(mnistCanvas.texture);
        Color *pixels = LoadImageColors(img);
        for (int i = 0; i < 784; i++) nnInput[i] = pixels[i].r / 255.0f;
        UnloadImageColors(pixels);
        UnloadImage(img);

        if (IsKeyPressed(KEY_ENTER)) predictedDigit = NeuralNet_Predict(nnInput);
        else if (IsKeyPressed(KEY_SPACE)) {
            BeginTextureMode(drawCanvas);
                ClearBackground(BLACK);
            EndTextureMode();
            predictedDigit = -1;
        }

        BeginDrawing();
            ClearBackground((Color){ 30, 30, 30, 255 });

            DrawText("Draw a digit", 20, 20, 20, WHITE);
            DrawText("ENTER = predict | SPACE = clear", 20, 45, 14, GRAY);

            DrawTextureRec(drawCanvas.texture, 
                  (Rectangle){0, 0, drawCanvas.texture.width, -drawCanvas.texture.height},
                  (Vector2){20, 80}, WHITE);
            DrawRectangleLines(20, 80, CANVAS_SIZE, CANVAS_SIZE, GRAY);

            DrawText("28x28 Converted Input", 340, 80, 20, WHITE);
            DrawTextureEx(mnistCanvas.texture, (Vector2){ 340, 120 }, 0, 5, WHITE);

            if (predictedDigit >= 0) {
                DrawText(
                    TextFormat("Prediction: %d", predictedDigit),
                    340, 280, 30, GREEN
                );
            }

        EndDrawing();
    }

    UnloadRenderTexture(drawCanvas);
    UnloadRenderTexture(mnistCanvas);
    CloseWindow();

    return 0;
}
