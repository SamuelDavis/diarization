<?php

[, $templatePath, $callsPath] = $argv;

$templatePath = realpath($templatePath);
$template = file_get_contents($templatePath);

$callsPath = realpath($callsPath);
$contents = file_get_contents($callsPath);
$calls = json_decode($contents);

foreach ($calls as $url => $milliseconds) {
    $seconds = $milliseconds / 1000;
    preg_match('/cra:(.+)$/', $url, $matches);
    $id = $matches[1] ?? null;
    if (! $id) {
        $data = compact('url', 'matches', 'id');
        throw new UnexpectedValueException(var_export($data, true));
    }
    $date = date('Y-m-dTH:i:s', $seconds);
    $name = "$date.$id.mp3";
    $command = str_replace(['$URL', '$NAME'], [$url, $name], $template);
    exec($command, $output, $code);
    if ($code !== 0) {
        exit(var_export($output));
    }
    break;
    sleep(1);
}
